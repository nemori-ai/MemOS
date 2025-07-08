import argparse
import asyncio
import json
import os

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import time

import pandas as pd

from dotenv import load_dotenv
from mem0 import MemoryClient
from tqdm import tqdm
from utils import filter_memory_data
from zep_cloud.client import Zep

from memos.configs.mem_os import MOSConfig
from memos.configs.memory import MemoryConfigFactory
from memos.mem_os.main import MOS
from memos.memories.factory import MemoryFactory


# Nemori imports
try:
    from nemori.retrieval import (
        RetrievalConfig,
        RetrievalQuery,
        RetrievalService,
        RetrievalStorageType,
        RetrievalStrategy,
    )
    from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
    from nemori.storage.storage_types import StorageConfig
    NEMORI_AVAILABLE = True
except ImportError:
    NEMORI_AVAILABLE = False
    print("⚠️ Nemori not available. Install nemori to use nemori functionality.")


async def get_nemori_client(user_id: str, version: str = "default"):
    """Get Nemori client for search."""
    if not NEMORI_AVAILABLE:
        raise ImportError("Nemori is not available. Please install nemori.")
    
    # Setup storage
    storage_dir = Path(f"results/locomo/nemori-{version}/storages")
    db_path = storage_dir / "nemori_memory.duckdb"
    
    if not db_path.exists():
        raise FileNotFoundError(f"Nemori database not found at {db_path}. Please run ingestion first.")
    
    storage_config = StorageConfig(
        backend_type="duckdb",
        connection_string=str(db_path),
        batch_size=100,
        cache_size=1000,
        enable_semantic_search=False,
    )
    
    episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
    await episode_repo.initialize()
    
    # Setup retrieval
    retrieval_service = RetrievalService(episode_repo)
    retrieval_config = RetrievalConfig(
        storage_type=RetrievalStorageType.DISK,
        storage_config={"directory": str(storage_dir)},
    )
    retrieval_service.register_provider(RetrievalStrategy.BM25, retrieval_config)
    await retrieval_service.initialize()
    
    return retrieval_service


def get_client(frame: str, user_id: str | None = None, version: str = "default"):
    if frame == "zep":
        zep = Zep(api_key=os.getenv("ZEP_API_KEY"), base_url="https://api.getzep.com/api/v2")
        return zep

    elif frame == "mem0" or frame == "mem0_graph":
        mem0 = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        return mem0

    elif frame == "memos":
        config_path = "configs/text_memos_config.json"
        with open(config_path) as f:
            config_data = json.load(f)
        config_data["config"]["extractor_llm"]["config"]["model_name_or_path"] = os.getenv("MODEL")
        config_data["config"]["extractor_llm"]["config"]["api_key"] = os.getenv("OPENAI_API_KEY")
        config_data["config"]["extractor_llm"]["config"]["api_base"] = os.getenv("OPENAI_BASE_URL")
        config_data["config"]["vector_db"]["config"]["path"] = (
            f"results/locomo/memos-{version}/storages/{user_id}/qdrant"
        )
        config_data["config"]["embedder"]["config"]["model_name_or_path"] = os.getenv(
            "EMBEDDING_MODEL"
        )

        config = MemoryConfigFactory.model_validate(config_data)

        m = MemoryFactory.from_config(config)
        m.load(f"results/locomo/memos-{version}/storages/{user_id}")
        return m

    elif frame == "memos_mos":
        mos_config_path = "configs/mos_memos_config.json"
        with open(mos_config_path) as f:
            mos_config_data = json.load(f)
        mos_config = MOSConfig(**mos_config_data)
        mos = MOS(mos_config)
        mos.create_user(user_id=user_id)

        storage_path = f"results/locomo/{frame}-{version}/storages/{user_id}"

        mos.register_mem_cube(
            mem_cube_name_or_path=storage_path,
            mem_cube_id=user_id,
            user_id=user_id,
        )

        return mos


TEMPLATE_ZEP = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts for the conversation along with the datetime of the event that the fact refers to.
If a fact mentions something happening a week ago, then the datetime will be the date time of last week and not the datetime
of when the fact was stated.
Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.

<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""

TEMPLATE_MEM0 = """Memories for user {speaker_1_user_id}:

    {speaker_1_memories}

    Memories for user {speaker_2_user_id}:

    {speaker_2_memories}
"""

TEMPLATE_MEM0_GRAPH = """Memories for user {speaker_1_user_id}:

    {speaker_1_memories}

    Relations for user {speaker_1_user_id}:

    {speaker_1_graph_memories}

    Memories for user {speaker_2_user_id}:

    {speaker_2_memories}

    Relations for user {speaker_2_user_id}:

    {speaker_2_graph_memories}
"""

TEMPLATE_MEMOS = """Memories for user {speaker_1}:

    {speaker_1_memories}

    Memories for user {speaker_2}:

    {speaker_2_memories}
"""

TEMPLATE_NEMORI = """Memories for user {speaker_1}:

    {speaker_1_memories}

    Memories for user {speaker_2}:

    {speaker_2_memories}
"""


def memos_search(client, query):
    start = time()
    search_results = client.search(query, top_k=20)
    context = ""
    for item in search_results:
        item = item.to_dict()
        context += f"{item['memory']}\n"
    print(query, context)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def mem0_search(client, query, speaker_a_user_id, speaker_b_user_id, top_k=20):
    start = time()
    search_speaker_a_results = client.search(
        query=query,
        top_k=top_k,
        user_id=speaker_a_user_id,
        output_format="v1.1",
        version="v2",
        filters={"AND": [{"user_id": f"{speaker_a_user_id}"}, {"run_id": "*"}]},
    )
    search_speaker_b_results = client.search(
        query=query,
        top_k=top_k,
        user_id=speaker_b_user_id,
        output_format="v1.1",
        version="v2",
        filters={"AND": [{"user_id": f"{speaker_b_user_id}"}, {"run_id": "*"}]},
    )

    search_speaker_a_memory = [
        {
            "memory": memory["memory"],
            "timestamp": memory["created_at"],
            "score": round(memory["score"], 2),
        }
        for memory in search_speaker_a_results["results"]
    ]

    search_speaker_a_memory = [
        [f"{item['timestamp']}: {item['memory']}" for item in search_speaker_a_memory]
    ]

    search_speaker_b_memory = [
        {
            "memory": memory["memory"],
            "timestamp": memory["created_at"],
            "score": round(memory["score"], 2),
        }
        for memory in search_speaker_b_results["results"]
    ]

    search_speaker_b_memory = [
        [f"{item['timestamp']}: {item['memory']}" for item in search_speaker_b_memory]
    ]

    context = TEMPLATE_MEM0.format(
        speaker_1_user_id=speaker_a_user_id.split("_")[0],
        speaker_1_memories=json.dumps(search_speaker_a_memory, indent=4),
        speaker_2_user_id=speaker_b_user_id.split("_")[0],
        speaker_2_memories=json.dumps(search_speaker_b_memory, indent=4),
    )

    print(query, context)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def memos_mos_search(client, query, conv_id, speaker_a, speaker_b, reversed_client=None):
    start = time()
    search_a_results = client.search(
        query=query,
        user_id=conv_id + "_speaker_a",
    )
    filtered_search_a_results = filter_memory_data(search_a_results)["text_mem"][0]["memories"]
    speaker_a_context = ""
    for item in filtered_search_a_results:
        speaker_a_context += f"{item['memory']}\n"

    search_b_results = reversed_client.search(
        query=query,
        user_id=conv_id + "_speaker_b",
    )
    filtered_search_b_results = filter_memory_data(search_b_results)["text_mem"][0]["memories"]
    speaker_b_context = ""
    for item in filtered_search_b_results:
        speaker_b_context += f"{item['memory']}\n"

    context = TEMPLATE_MEMOS.format(
        speaker_1=speaker_a,
        speaker_1_memories=speaker_a_context,
        speaker_2=speaker_b,
        speaker_2_memories=speaker_b_context,
    )

    print(query, context)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def mem0_graph_search(client, query, speaker_a_user_id, speaker_b_user_id, top_k=20):
    start = time()
    search_speaker_a_results = client.search(
        query=query,
        top_k=top_k,
        user_id=speaker_a_user_id,
        output_format="v1.1",
        version="v2",
        enable_graph=True,
        filters={"AND": [{"user_id": f"{speaker_a_user_id}"}, {"run_id": "*"}]},
    )
    search_speaker_b_results = client.search(
        query=query,
        top_k=top_k,
        user_id=speaker_b_user_id,
        output_format="v1.1",
        version="v2",
        enable_graph=True,
        filters={"AND": [{"user_id": f"{speaker_b_user_id}"}, {"run_id": "*"}]},
    )

    search_speaker_a_memory = [
        {
            "memory": memory["memory"],
            "timestamp": memory["created_at"],
            "score": round(memory["score"], 2),
        }
        for memory in search_speaker_a_results["results"]
    ]

    search_speaker_a_memory = [
        [f"{item['timestamp']}: {item['memory']}" for item in search_speaker_a_memory]
    ]

    search_speaker_b_memory = [
        {
            "memory": memory["memory"],
            "timestamp": memory["created_at"],
            "score": round(memory["score"], 2),
        }
        for memory in search_speaker_b_results["results"]
    ]

    search_speaker_b_memory = [
        [f"{item['timestamp']}: {item['memory']}" for item in search_speaker_b_memory]
    ]

    search_speaker_a_graph = [
        {
            "source": relation["source"],
            "relationship": relation["relationship"],
            "target": relation["target"],
        }
        for relation in search_speaker_a_results["relations"]
    ]

    search_speaker_b_graph = [
        {
            "source": relation["source"],
            "relationship": relation["relationship"],
            "target": relation["target"],
        }
        for relation in search_speaker_b_results["relations"]
    ]
    context = TEMPLATE_MEM0_GRAPH.format(
        speaker_1_user_id=speaker_a_user_id.split("_")[0],
        speaker_1_memories=json.dumps(search_speaker_a_memory, indent=4),
        speaker_1_graph_memories=json.dumps(search_speaker_a_graph, indent=4),
        speaker_2_user_id=speaker_b_user_id.split("_")[0],
        speaker_2_memories=json.dumps(search_speaker_b_memory, indent=4),
        speaker_2_graph_memories=json.dumps(search_speaker_b_graph, indent=4),
    )
    print(query, context)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def zep_search(client, query, group_id, top_k=20):
    start = time()
    nodes_result = client.graph.search(
        query=query,
        group_id=group_id,
        scope="nodes",
        reranker="rrf",
        limit=top_k,
    )
    edges_result = client.graph.search(
        query=query,
        group_id=group_id,
        scope="edges",
        reranker="cross_encoder",
        limit=top_k,
    )

    nodes = nodes_result.nodes
    edges = edges_result.edges

    facts = [f"  - {edge.fact} (event_time: {edge.valid_at})" for edge in edges]
    entities = [f"  - {node.name}: {node.summary}" for node in nodes]
    context = TEMPLATE_ZEP.format(facts="\n".join(facts), entities="\n".join(entities))

    duration_ms = (time() - start) * 1000

    return context, duration_ms


async def nemori_search(retrieval_service, query, speaker_a_user_id, speaker_b_user_id, top_k=20):
    """Search using Nemori."""
    start = time()
    
    print(f"\n🔍 [NEMORI SEARCH] Starting search for query: '{query}'")
    print(f"   👤 Speaker A ID: '{speaker_a_user_id}'")
    print(f"   👤 Speaker B ID: '{speaker_b_user_id}'")
    print(f"   📊 Top K: {top_k}")
    
    # Search for speaker A
    print(f"\n🔎 [SPEAKER A] Searching for owner_id: '{speaker_a_user_id}'")
    query_a = RetrievalQuery(text=query, owner_id=speaker_a_user_id, limit=top_k, strategy=RetrievalStrategy.BM25)
    print(f"   📝 Query object: text='{query_a.text}', owner_id='{query_a.owner_id}', limit={query_a.limit}")
    
    try:
        result_a = await retrieval_service.search(query_a)
        print(f"   ✅ Search completed. Found {len(result_a.episodes)} episodes")
        
        if len(result_a.episodes) > 0:
            print("   📋 Sample episodes for speaker A:")
            for i, episode in enumerate(result_a.episodes[:2]):
                print(f"     {i+1}. Title: '{episode.title}'")
                print(f"        Content: '{episode.content[:100]}...'")
                print(f"        Summary: '{episode.summary}'")
        else:
            print("   ⚠️ No episodes found for speaker A")
            
    except Exception as e:
        print(f"   ❌ Search failed for speaker A: {e}")
        result_a = type('obj', (object,), {'episodes': []})()
    
    # Search for speaker B
    print(f"\n🔎 [SPEAKER B] Searching for owner_id: '{speaker_b_user_id}'")
    query_b = RetrievalQuery(text=query, owner_id=speaker_b_user_id, limit=top_k, strategy=RetrievalStrategy.BM25)
    print(f"   📝 Query object: text='{query_b.text}', owner_id='{query_b.owner_id}', limit={query_b.limit}")
    
    try:
        result_b = await retrieval_service.search(query_b)
        print(f"   ✅ Search completed. Found {len(result_b.episodes)} episodes")
        
        if len(result_b.episodes) > 0:
            print("   📋 Sample episodes for speaker B:")
            for i, episode in enumerate(result_b.episodes[:2]):
                print(f"     {i+1}. Title: '{episode.title}'")
                print(f"        Content: '{episode.content[:100]}...'")
                print(f"        Summary: '{episode.summary}'")
        else:
            print("   ⚠️ No episodes found for speaker B")
            
    except Exception as e:
        print(f"   ❌ Search failed for speaker B: {e}")
        result_b = type('obj', (object,), {'episodes': []})()
    
    # Format results for speaker A
    speaker_a_memories = []
    for episode in result_a.episodes:
        memory_text = f"{episode.title}: {episode.content}"
        speaker_a_memories.append(memory_text)
    
    # Format results for speaker B
    speaker_b_memories = []
    for episode in result_b.episodes:
        memory_text = f"{episode.title}: {episode.content}"
        speaker_b_memories.append(memory_text)
    
    print(f"\n📊 [FORMATTING] Speaker A memories: {len(speaker_a_memories)}")
    print(f"📊 [FORMATTING] Speaker B memories: {len(speaker_b_memories)}")
    
    # Format context
    context = TEMPLATE_NEMORI.format(
        speaker_1=speaker_a_user_id.split("_")[0] if "_" in speaker_a_user_id else speaker_a_user_id,
        speaker_1_memories="\n".join(speaker_a_memories) if speaker_a_memories else "No relevant memories found",
        speaker_2=speaker_b_user_id.split("_")[0] if "_" in speaker_b_user_id else speaker_b_user_id,
        speaker_2_memories="\n".join(speaker_b_memories) if speaker_b_memories else "No relevant memories found",
    )
    
    print("\n📄 [CONTEXT] Generated context preview:")
    print(f"   {context[:200]}...")
    
    duration_ms = (time() - start) * 1000
    print(f"\n⏱️ [TIMING] Search completed in {duration_ms:.2f}ms")
    
    return context, duration_ms


async def search_query_async(client, query, metadata, frame, reversed_client=None, top_k=20):
    """Async version of search_query for nemori."""
    speaker_a_user_id = metadata.get("speaker_a_user_id")
    speaker_b_user_id = metadata.get("speaker_b_user_id")
    
    if frame == "nemori":
        context, duration_ms = await nemori_search(
            client, query, speaker_a_user_id, speaker_b_user_id, top_k
        )
        return context, duration_ms
    else:
        # For non-async frameworks, call the sync version
        return search_query(client, query, metadata, frame, reversed_client, top_k)


def search_query(client, query, metadata, frame, reversed_client=None, top_k=20):
    conv_id = metadata.get("conv_id")
    speaker_a = metadata.get("speaker_a")
    speaker_b = metadata.get("speaker_b")
    speaker_a_user_id = metadata.get("speaker_a_user_id")
    speaker_b_user_id = metadata.get("speaker_b_user_id")

    if frame == "zep":
        context, duration_ms = zep_search(client, query, conv_id, top_k)
    elif frame == "memos":
        context, duration_ms = memos_search(client, query)
    elif frame == "mem0":
        context, duration_ms = mem0_search(
            client, query, speaker_a_user_id, speaker_b_user_id, top_k
        )
    elif frame == "mem0_graph":
        context, duration_ms = mem0_graph_search(
            client, query, speaker_a_user_id, speaker_b_user_id, top_k
        )
    elif frame == "memos_mos":
        context, duration_ms = memos_mos_search(
            client, query, conv_id, speaker_a, speaker_b, reversed_client
        )
    return context, duration_ms


def load_existing_results(frame, version, group_idx):
    result_path = (
        f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{group_idx}.json"
    )
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                return json.load(f), True
        except Exception as e:
            print(f"Error loading existing results for group {group_idx}: {e}")
    return {}, False


async def process_user_nemori(group_idx, locomo_df, frame, version, top_k=20):
    """Process user for Nemori framework."""
    print(f"\n🚀 [NEMORI PROCESS] Starting processing for user {group_idx}")
    
    search_results = defaultdict(list)
    qa_set = locomo_df["qa"].iloc[group_idx]
    conversation = locomo_df["conversation"].iloc[group_idx]
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    speaker_a_user_id = f"{speaker_a.lower().replace(' ', '_')}_{group_idx}"
    speaker_b_user_id = f"{speaker_b.lower().replace(' ', '_')}_{group_idx}"
    conv_id = f"locomo_exp_user_{group_idx}"
    
    print(f"   👥 Original speakers: '{speaker_a}' & '{speaker_b}'")
    print(f"   🆔 Generated IDs: '{speaker_a_user_id}' & '{speaker_b_user_id}'")
    print(f"   📝 Conversation ID: '{conv_id}'")
    print(f"   ❓ QA set size: {len(qa_set)}")
    
    existing_results, loaded = load_existing_results(frame, version, group_idx)
    if loaded:
        print(f"Loaded existing results for group {group_idx}")
        return existing_results
    
    metadata = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "speaker_a_user_id": speaker_a_user_id,
        "speaker_b_user_id": speaker_b_user_id,
        "conv_idx": group_idx,
        "conv_id": conv_id,
    }
    
    # Get nemori client
    client = await get_nemori_client(conv_id, version)
    
    async def process_qa(qa):
        query = qa.get("question")
        if qa.get("category") == 5:
            return None
        
        context, duration_ms = await search_query_async(
            client, query, metadata, frame, top_k=top_k
        )
        
        if not context:
            print(f"No context found for query: {query}")
            context = ""
        return {"query": query, "context": context, "duration_ms": duration_ms}
    
    # Process QAs sequentially for nemori (since it's async)
    for qa in tqdm(qa_set, desc=f"Processing user {group_idx}"):
        result = await process_qa(qa)
        if result:
            context_preview = (
                result["context"][:20] + "..." if result["context"] else "No context"
            )
            print(
                {
                    "query": result["query"],
                    "context": context_preview,
                    "duration_ms": result["duration_ms"],
                }
            )
            search_results[conv_id].append(result)
    
    # Cleanup
    if hasattr(client, 'close'):
        await client.close()
    
    os.makedirs(f"results/locomo/{frame}-{version}/tmp/", exist_ok=True)
    with open(
        f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{group_idx}.json", "w"
    ) as f:
        json.dump(dict(search_results), f, indent=2)
        print(f"Save search results {group_idx}")
    
    return search_results


def process_user(group_idx, locomo_df, frame, version, top_k=20, num_workers=1):
    search_results = defaultdict(list)
    qa_set = locomo_df["qa"].iloc[group_idx]
    conversation = locomo_df["conversation"].iloc[group_idx]
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    speaker_a_user_id = f"{speaker_a.lower().replace(' ', '_')}_{group_idx}"
    speaker_b_user_id = f"{speaker_b.lower().replace(' ', '_')}_{group_idx}"
    conv_id = f"locomo_exp_user_{group_idx}"

    existing_results, loaded = load_existing_results(frame, version, group_idx)
    if loaded:
        print(f"Loaded existing results for group {group_idx}")
        return existing_results

    metadata = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "speaker_a_user_id": speaker_a_user_id,
        "speaker_b_user_id": speaker_b_user_id,
        "conv_idx": group_idx,
        "conv_id": conv_id,
    }

    reversed_client = None
    if frame == "memos_mos":
        speaker_a_user_id = conv_id + "_speaker_a"
        speaker_b_user_id = conv_id + "_speaker_b"
        client = get_client(frame, speaker_a_user_id, version)
        reversed_client = get_client(frame, speaker_b_user_id, version)
    else:
        client = get_client(frame, conv_id, version)

    def process_qa(qa):
        query = qa.get("question")
        if qa.get("category") == 5:
            return None
        context, duration_ms = search_query(
            client, query, metadata, frame, reversed_client=reversed_client, top_k=top_k
        )

        if not context:
            print(f"No context found for query: {query}")
            context = ""
        return {"query": query, "context": context, "duration_ms": duration_ms}

    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for qa in qa_set:
            futures.append(executor.submit(process_qa, qa))

        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Processing user {group_idx}"
        ):
            result = future.result()
            if result:
                context_preview = (
                    result["context"][:20] + "..." if result["context"] else "No context"
                )
                print(
                    {
                        "query": result["query"],
                        "context": context_preview,
                        "duration_ms": result["duration_ms"],
                    }
                )
                search_results[conv_id].append(result)

    os.makedirs(f"results/locomo/{frame}-{version}/tmp/", exist_ok=True)
    with open(
        f"results/locomo/{frame}-{version}/tmp/{frame}_locomo_search_results_{group_idx}.json", "w"
    ) as f:
        json.dump(dict(search_results), f, indent=2)
        print(f"Save search results {group_idx}")

    return search_results


async def main_nemori(version="default", top_k=20):
    """Main function for Nemori search."""
    load_dotenv()
    locomo_df = pd.read_json("data/locomo/locomo10.json")

    num_conv = 10
    frame = "nemori"
    os.makedirs(f"results/locomo/{frame}-{version}/", exist_ok=True)
    all_search_results = defaultdict(list)

    for idx in range(num_conv):
        try:
            print(f"Processing user {idx}...")
            user_results = await process_user_nemori(idx, locomo_df, frame, version, top_k)
            for conv_id, results in user_results.items():
                all_search_results[conv_id].extend(results)
        except Exception as e:
            print(f"User {idx} generated an exception: {e}")

    with open(f"results/locomo/{frame}-{version}/{frame}_locomo_search_results.json", "w") as f:
        json.dump(dict(all_search_results), f, indent=2)
        print("Save all search results")


def main(frame, version="default", num_workers=1, top_k=20):
    load_dotenv()
    
    if frame == "nemori":
        if not NEMORI_AVAILABLE:
            print("❌ Nemori is not available. Please install nemori to use this framework.")
            return
        # Run async main for nemori
        asyncio.run(main_nemori(version, top_k))
        return
    
    locomo_df = pd.read_json("data/locomo/locomo10.json")

    num_users = 10
    os.makedirs(f"results/locomo/{frame}-{version}/", exist_ok=True)
    all_search_results = defaultdict(list)

    for idx in range(num_users):
        try:
            print(f"Processing user {idx}...")
            user_results = process_user(idx, locomo_df, frame, version, top_k, num_workers)
            for conv_id, results in user_results.items():
                all_search_results[conv_id].extend(results)
        except Exception as e:
            print(f"User {idx} generated an exception: {e}")

    with open(f"results/locomo/{frame}-{version}/{frame}_locomo_search_results.json", "w") as f:
        json.dump(dict(all_search_results), f, indent=2)
        print("Save all search results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        choices=["zep", "memos", "mem0", "mem0_graph", "memos_mos", "langmem", "nemori"],
        help="Specify the memory framework (zep or memos or mem0 or mem0_graph or memos_mos or nemori)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for saving results (e.g., 1010)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers to process users"
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Number of results to retrieve in search queries"
    )
    args = parser.parse_args()
    lib = args.lib
    version = args.version
    workers = args.workers
    top_k = args.top_k

    main(lib, version, workers, top_k)
