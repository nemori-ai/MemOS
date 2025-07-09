import argparse
import asyncio
import concurrent.futures
import json
import os
import time

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
from mem0 import MemoryClient
from tqdm import tqdm
from zep_cloud.client import Zep

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.configs.memory import MemoryConfigFactory
from memos.mem_cube.general import GeneralMemCube
from memos.mem_os.main import MOS
from memos.memories.factory import MemoryFactory


# Nemori imports
try:
    from nemori.builders.conversation_builder import ConversationEpisodeBuilder
    from nemori.core.builders import EpisodeBuilderRegistry
    from nemori.core.data_types import ConversationData, DataType, RawEventData, TemporalInfo
    from nemori.episode_manager import EpisodeManager
    from nemori.llm.providers.openai_provider import OpenAIProvider
    from nemori.retrieval import (
        RetrievalConfig,
        RetrievalService,
        RetrievalStorageType,
        RetrievalStrategy,
    )
    from nemori.storage.duckdb_storage import (
        DuckDBEpisodicMemoryRepository,
        DuckDBRawDataRepository,
    )
    from nemori.storage.storage_types import StorageConfig

    NEMORI_AVAILABLE = True
except ImportError:
    NEMORI_AVAILABLE = False
    print("⚠️ Nemori not available. Install nemori to use nemori functionality.")


custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


class NemoriExperiment:
    """Nemori experiment adapted from LoCoMoExperiment."""

    def __init__(
        self, version: str = "default", episode_mode: str = "speaker", max_concurrency: int = 3
    ):
        self.version = version
        self.episode_mode = episode_mode
        self.max_concurrency = max_concurrency

        # Paths
        self.db_dir = Path(f"results/locomo/nemori-{version}/storages")
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.raw_data_repo = None
        self.episode_repo = None
        self.retrieval_service = None
        self.episode_manager = None
        self.llm_provider = None

        # Data
        self.conversations = []
        self.episodes = []

    async def setup_llm_provider(self) -> bool:
        """Setup OpenAI LLM provider if API key is available."""
        print("\n🤖 Setting up LLM Provider")
        print("=" * 50)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ OPENAI_API_KEY not found in environment")
            return False

        try:
            self.llm_provider = OpenAIProvider(
                model="gpt-4o-mini", temperature=0.3, max_tokens=16 * 1024
            )

            if await self.llm_provider.test_connection():
                print("✅ OpenAI connection successful!")
                print(f"🎯 Model: {self.llm_provider.model}")
                return True
            else:
                print("❌ OpenAI connection failed!")
                return False
        except Exception as e:
            print(f"❌ Error setting up OpenAI provider: {e}")
            return False

    async def setup_storage_and_retrieval(self):
        """Setup DuckDB storage and BM25 retrieval service."""
        print("\n🗄️ Setting up Storage and Retrieval")
        print("=" * 50)

        # Setup DuckDB storage
        db_path = self.db_dir / "nemori_memory.duckdb"

        # Remove existing database to start fresh
        if db_path.exists():
            db_path.unlink()
            print("🧹 Cleaned existing database")

        # Also clean any existing BM25 indices
        for index_file in self.db_dir.glob("bm25_index_*.pkl"):
            index_file.unlink()
            print(f"🧹 Cleaned existing index: {index_file.name}")

        # Create storage configurations
        storage_config = StorageConfig(
            backend_type="duckdb",
            connection_string=str(db_path),
            batch_size=100,
            cache_size=1000,
            enable_semantic_search=False,
        )

        # Initialize repositories
        self.raw_data_repo = DuckDBRawDataRepository(storage_config)
        self.episode_repo = DuckDBEpisodicMemoryRepository(storage_config)

        await self.raw_data_repo.initialize()
        await self.episode_repo.initialize()

        print(f"✅ DuckDB storage initialized: {db_path}")

        # Setup BM25 retrieval service
        self.retrieval_service = RetrievalService(self.episode_repo)

        # Create BM25 retrieval provider configuration with disk storage
        retrieval_config = RetrievalConfig(
            storage_type=RetrievalStorageType.DISK,
            storage_config={"directory": str(self.db_dir)},
        )

        # Register the provider with the service
        self.retrieval_service.register_provider(RetrievalStrategy.BM25, retrieval_config)

        # Initialize the retrieval service
        await self.retrieval_service.initialize()

        print("✅ BM25 retrieval service configured")

        # Setup episode manager
        builder_registry = EpisodeBuilderRegistry()
        if self.llm_provider:
            conversation_builder = ConversationEpisodeBuilder(llm_provider=self.llm_provider)
        else:
            raise ValueError("LLM provider not found")

        builder_registry.register(conversation_builder)

        self.episode_manager = EpisodeManager(
            raw_data_repo=self.raw_data_repo,
            episode_repo=self.episode_repo,
            builder_registry=builder_registry,
            retrieval_service=self.retrieval_service,
        )

        print("✅ Episode manager initialized")

    def load_locomo_data(self, locomo_df):
        """Load LoComo data from DataFrame."""
        self.conversations = [locomo_df.iloc[i].to_dict() for i in range(len(locomo_df))]
        print(f"📊 Loaded {len(self.conversations)} conversations")

    def parse_locomo_timestamp(self, timestamp_str: str) -> datetime:
        """Parse LoComo timestamp format to datetime object."""
        try:
            timestamp_str = timestamp_str.replace("\\s+", " ").strip()
            dt = datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
            return dt
        except ValueError as e:
            print(f"⚠️ Warning: Could not parse timestamp '{timestamp_str}': {e}")
            return datetime.now()

    def convert_locomo_to_nemori(
        self, conversation_data: dict, conversation_id: str
    ) -> RawEventData:
        """Convert LoComo conversation format to Nemori RawEventData format."""
        print(f"   🔄 Converting LoComo conversation {conversation_id}...")
        messages = []
        conv = conversation_data["conversation"]

        # Get all session keys in order
        session_keys = sorted(
            [
                key
                for key in conv
                if key.startswith("session_") and not key.endswith("_date_time")
            ]
        )

        print(f"   📅 Found {len(session_keys)} sessions")
        print(
            f"   🎭 Speakers: {conv.get('speaker_a', 'Unknown')} & {conv.get('speaker_b', 'Unknown')}"
        )

        # Generate unique speaker IDs for this conversation
        speaker_name_to_id = {}
        for session_key in session_keys:
            session_messages = conv[session_key]
            session_time_key = f"{session_key}_date_time"

            if session_time_key in conv:
                # Parse session timestamp
                session_time = self.parse_locomo_timestamp(conv[session_time_key])

                # Process each message in this session
                for i, msg in enumerate(session_messages):
                    # Generate timestamp for this message (session time + message offset)
                    msg_timestamp = session_time + timedelta(
                        seconds=i * 30
                    )  # 30 seconds between messages
                    iso_timestamp = msg_timestamp.isoformat()

                    # Generate unique speaker_id for this conversation
                    speaker_name = msg["speaker"]
                    if speaker_name not in speaker_name_to_id:
                        # Generate unique ID: {name}_{conversation_index}
                        unique_id = f"{speaker_name.lower().replace(' ', '_')}_{conversation_id}"
                        speaker_name_to_id[speaker_name] = unique_id

                    # Process content with image information if present
                    content = msg["text"]
                    if "img_url" in msg and msg["img_url"]:
                        blip_caption = msg.get("blip_caption", "an image")
                        content = f"[{speaker_name} shared an image: {blip_caption}] {content}"

                    message = {
                        "speaker_id": speaker_name_to_id[speaker_name],
                        "user_name": speaker_name,
                        "content": content,
                        "timestamp": iso_timestamp,
                        "original_timestamp": conv[session_time_key],
                        "dia_id": msg["dia_id"],
                        "session": session_key,
                    }

                    # Add optional fields if present
                    for optional_field in ["img_url", "blip_caption", "query"]:
                        if optional_field in msg:
                            message[optional_field] = msg[optional_field]

                    messages.append(message)

        print(f"   ✅ Converted {len(messages)} messages from {len(session_keys)} sessions")

        # Calculate total duration based on session lengths
        if messages:
            first_time = datetime.fromisoformat(messages[0]["timestamp"])

            # Calculate duration as total session time rather than span across all sessions
            session_durations = {}
            for msg in messages:
                session = msg["session"]
                if session not in session_durations:
                    session_durations[session] = 0
                session_durations[session] += 1  # Count messages per session

            # Estimate duration: 30 seconds per message + 5 minutes setup per session
            total_duration = sum(msg_count * 30 + 300 for msg_count in session_durations.values())
            duration = total_duration
        else:
            duration = 0.0
            first_time = datetime.now()

        temporal_info = TemporalInfo(timestamp=first_time, duration=duration, timezone="UTC")

        return RawEventData(
            data_type=DataType.CONVERSATION,
            content=messages,
            source="locomo_dataset",
            temporal_info=temporal_info,
            metadata={
                "conversation_id": conversation_id,
                "sample_id": conversation_data.get("sample_id", "unknown"),
                "speaker_a": conv.get("speaker_a"),
                "speaker_b": conv.get("speaker_b"),
                "participant_count": 2,
                "session_count": len(session_keys),
                "message_count": len(messages),
                "has_images": any("img_url" in msg for msg in messages),
                "original_format": "locomo_multi_session",
                "episode_mode": self.episode_mode,
            },
        )

    async def _detect_conversation_boundaries(self, messages: list) -> list[tuple[int, int, str]]:
        """Detect conversation boundaries using the conversation builder's boundary detection."""
        print(f"\n     🔍 Starting boundary detection for {len(messages)} messages")

        boundaries = [(0, len(messages) - 1, "Single episode - no boundary detection")]  # Default: single episode

        if not self.llm_provider or len(messages) <= 1:
            print("     ⚠️ No LLM provider or too few messages, using single episode")
            return boundaries

        print("     ⚠️ Note: Boundary detection uses sequential LLM calls")

        # Create a conversation builder for boundary detection
        builder = ConversationEpisodeBuilder(llm_provider=self.llm_provider)

        # Convert messages to the format expected by boundary detection
        message_dicts = []
        for msg in messages:
            message_dict = {
                "content": msg.content,
                "speaker_id": msg.speaker_id,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            }
            message_dicts.append(message_dict)

        print("\n     🔍 Starting boundary detection analysis...")

        # Detect boundaries by checking each message against conversation history
        boundaries = []
        current_start = 0
        current_episode_reason = "Episode start"

        for i in range(1, len(message_dicts)):
            # Check if we should end the current episode at this message
            current_episode_history = message_dicts[current_start:i]
            new_message = message_dicts[i]

            # Use async boundary detection
            should_end, reason = await builder._detect_boundary(
                conversation_history=current_episode_history, new_messages=[new_message]
            )

            if should_end:
                # End current episode and start new one
                boundaries.append((current_start, i - 1, current_episode_reason))
                print(f"     ✂️ Boundary at message {i}: {reason}")
                current_start = i
                current_episode_reason = reason  # The reason becomes the context for the next episode

        # Add the final segment
        boundaries.append((current_start, len(message_dicts) - 1, current_episode_reason))

        print(f"     📊 Detected {len(boundaries)} conversation segments")

        return boundaries

    def _calculate_segment_duration(self, messages: list) -> float:
        """Calculate the duration of a message segment."""
        if len(messages) < 2:
            return 300.0  # Default 5 minutes for single message

        first_msg = messages[0]
        last_msg = messages[-1]

        if first_msg.timestamp and last_msg.timestamp:
            duration = (last_msg.timestamp - first_msg.timestamp).total_seconds()
            return max(duration, 60.0)  # Minimum 1 minute
        else:
            # Estimate based on message count (30 seconds per message)
            return len(messages) * 30.0

    async def _build_episodes_for_speaker(
        self, raw_data: RawEventData, owner_id: str, episode_boundaries: list[tuple[int, int, str]]
    ) -> list:
        """Build episodes for a specific speaker using pre-detected boundaries."""
        conversation_data = ConversationData(raw_data)
        messages = conversation_data.messages
        episodes = []

        if not messages:
            return episodes

        # Create episodes for each boundary segment
        for start_idx, end_idx, boundary_reason in episode_boundaries:
            segment_messages = messages[start_idx : end_idx + 1]

            # Create a new RawEventData for this segment
            segment_raw_data = RawEventData(
                data_type=DataType.CONVERSATION,
                content=[
                    {
                        "speaker_id": msg.speaker_id,
                        "user_name": msg.user_name,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                        "metadata": msg.metadata,
                    }
                    for msg in segment_messages
                ],
                source=raw_data.source,
                temporal_info=TemporalInfo(
                    timestamp=segment_messages[0].timestamp or raw_data.temporal_info.timestamp,
                    duration=self._calculate_segment_duration(segment_messages),
                    timezone=raw_data.temporal_info.timezone,
                ),
                metadata={
                    **raw_data.metadata,
                    "segment_start": start_idx,
                    "segment_end": end_idx,
                    "total_segments": len(episode_boundaries),
                    "owner_id": owner_id,
                    "boundary_reason": boundary_reason,
                },
            )

            # Process the segment through episode manager
            episode = await self.episode_manager.process_raw_data(segment_raw_data, owner_id)
            if episode:
                episodes.append(episode)

        return episodes

    async def _build_episodes_speaker_mode(self, raw_data: RawEventData) -> list:
        """Build episodes using speaker perspective (each speaker gets their own episodes)."""
        # Get unique speakers from the conversation
        speakers = {
            msg["speaker_id"]
            for msg in raw_data.content
            if isinstance(msg, dict) and "speaker_id" in msg
        }

        print(f"   👥 Speakers: {list(speakers)}")
        print(f"   💬 Messages: {len(raw_data.content)}")
        print(f"   🕐 Duration: {raw_data.temporal_info.duration:.0f} seconds")

        # Detect conversation boundaries once for all speakers
        conversation_data = ConversationData(raw_data)
        episode_boundaries = await self._detect_conversation_boundaries(conversation_data.messages)

        all_episodes = []

        # Process speakers sequentially (only 2 speakers, concurrency not needed)
        for speaker_id in speakers:
            episodes = await self._build_episodes_for_speaker(
                raw_data, speaker_id, episode_boundaries
            )
            all_episodes.extend(episodes)
            print(f"   ✅ {speaker_id}: {len(episodes)} episodes")

        print(f"   📊 Total: {len(all_episodes)} episodes from {len(speakers)} speakers")
        return all_episodes

    async def build_episodes(self):
        """Build episodes from LoComo conversations using boundary detection."""
        print("\n🏗️ Building Episodes with Boundary Detection")
        print("=" * 50)
        print(f"🎭 Mode: {self.episode_mode} perspective")
        print(
            f"🔄 Processing {len(self.conversations)} conversations with max concurrency: {self.max_concurrency}"
        )

        self.episodes = []

        async def process_conversation(conv_index: int, conv_data: dict) -> tuple[str, list]:
            """Process a single conversation with concurrency control."""
            conv_id = str(conv_index)

            start_time = time.time()
            print(f"   🚀 Starting conversation {conv_id} processing... [{start_time:.1f}]")
            try:
                # Convert to Nemori format (can be concurrent)
                raw_data = self.convert_locomo_to_nemori(conv_data, conv_id)

                # Use speaker perspective
                episodes = await self._build_episodes_speaker_mode(raw_data)

                end_time = time.time()
                duration = end_time - start_time
                print(
                    f"   ✅ Conversation {conv_id}: {len(episodes)} episodes [{end_time:.1f}, took {duration:.1f}s]"
                )
                return conv_id, episodes

            except Exception as e:
                print(f"   ❌ Error processing conversation {conv_id}: {e}")
                return conv_id, []

        # Create tasks for all conversations
        tasks = [
            process_conversation(i, conv_data) for i, conv_data in enumerate(self.conversations)
        ]
        print(f"   📋 Created {len(tasks)} concurrent tasks")

        # Wait for all tasks to complete
        print("   ⏳ Starting concurrent execution...")
        results = await asyncio.gather(*tasks)
        print("   🏁 All conversations completed")

        # Collect all episodes
        for _, episodes in results:
            self.episodes.extend(episodes)

        print("\n📊 Episode Building Complete")
        print(f"✅ Successfully created {len(self.episodes)} episodes")

        # Build BM25 indices for all episodes
        await self.build_bm25_indices()

    async def build_bm25_indices(self):
        """Build BM25 indices for all episodes after they are created."""
        print("\n🔧 Building BM25 Indices")
        print("=" * 50)

        if not self.episodes:
            print("⚠️ No episodes to index")
            return

        # Get all unique owner_ids from episodes
        owner_ids = {episode.owner_id for episode in self.episodes}
        print(f"🎯 Building indices for {len(owner_ids)} owners: {list(owner_ids)}")

        # Force refresh of BM25 indices
        try:
            bm25_provider = self.retrieval_service.get_provider(RetrievalStrategy.BM25)
            if bm25_provider:
                print("🔄 Triggering BM25 index rebuild...")

                # Get all episodes from repository
                all_episodes = []
                for owner_id in owner_ids:
                    try:
                        result = await self.episode_repo.get_episodes_by_owner(owner_id)
                        # Handle EpisodeSearchResult object
                        owner_episodes = result.episodes if hasattr(result, "episodes") else result
                        all_episodes.extend(owner_episodes)
                        print(f"   📊 Owner {owner_id}: {len(owner_episodes)} episodes")
                    except Exception as e:
                        print(f"   ❌ Error getting episodes for {owner_id}: {e}")

                if all_episodes:
                    print(f"🏗️ Rebuilding indices for {len(all_episodes)} total episodes...")

                    # Trigger BM25 index building by performing dummy searches for each owner
                    for owner_id in owner_ids:
                        try:
                            result = await self.episode_repo.get_episodes_by_owner(owner_id)
                            # Handle EpisodeSearchResult object
                            if hasattr(result, "episodes"):
                                owner_episodes = result.episodes
                            else:
                                owner_episodes = result

                            if owner_episodes:
                                # Trigger index building by performing a search
                                from nemori.retrieval import RetrievalQuery

                                dummy_query = RetrievalQuery(
                                    text=".",
                                    owner_id=owner_id,
                                    limit=1,
                                    strategy=RetrievalStrategy.BM25,
                                )
                                await self.retrieval_service.search(dummy_query)
                                print(
                                    f"   ✅ Triggered index build for {owner_id}: {len(owner_episodes)} episodes"
                                )
                            else:
                                print(f"   ⚠️ No episodes found for {owner_id}")
                        except Exception as e:
                            print(f"   ❌ Error triggering index for {owner_id}: {e}")

                print("✅ BM25 index building completed")
            else:
                print("❌ BM25 provider not found")

        except Exception as e:
            print(f"❌ Error during BM25 index building: {e}")
            import traceback

            traceback.print_exc()

    async def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up")

        if self.retrieval_service:
            await self.retrieval_service.close()
        if self.raw_data_repo:
            await self.raw_data_repo.close()
        if self.episode_repo:
            await self.episode_repo.close()

        print("✅ Cleanup complete")


def get_client(frame: str, user_id: str | None = None, version: str = "default"):
    if frame == "zep":
        zep = Zep(api_key=os.getenv("ZEP_API_KEY"), base_url="https://api.getzep.com/api/v2")
        return zep

    elif frame == "mem0" or frame == "mem0_graph":
        mem0 = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        mem0.update_project(custom_instructions=custom_instructions)
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

        mem_cube_config_path = "configs/mem_cube_config.json"
        with open(mem_cube_config_path) as f:
            mem_cube_config_data = json.load(f)
        mem_cube_config_data["user_id"] = user_id
        mem_cube_config_data["cube_id"] = user_id
        mem_cube_config_data["text_mem"]["config"]["graph_db"]["config"]["db_name"] = (
            f"{user_id.replace('_', '')}{version}"
        )
        mem_cube_config = GeneralMemCubeConfig.model_validate(mem_cube_config_data)
        mem_cube = GeneralMemCube(mem_cube_config)

        storage_path = f"results/locomo/{frame}-{version}/storages/{user_id}"
        try:
            mem_cube.dump(storage_path)
        except Exception as e:
            print(f"dumping memory cube: {e!s} already exists, will use it")

        mos.register_mem_cube(
            mem_cube_name_or_path=storage_path,
            mem_cube_id=user_id,
            user_id=user_id,
        )

        return mos


def ingest_session(client, session, frame, metadata, revised_client=None):
    session_date = metadata["session_date"]
    date_format = "%I:%M %p on %d %B, %Y UTC"
    date_string = datetime.strptime(session_date, date_format).replace(tzinfo=timezone.utc)
    iso_date = date_string.isoformat()
    conv_idx = metadata["conv_idx"]
    conv_id = "locomo_exp_user_" + str(conv_idx)
    dt = datetime.fromisoformat(iso_date)
    timestamp = int(dt.timestamp())
    print(f"Processing conv {conv_id}, session {metadata['session_key']}")
    start_time = time.time()

    if frame == "zep":
        for chat in tqdm(session, desc=f"{metadata['session_key']}"):
            data = chat.get("speaker") + ": " + chat.get("text")
            print({"context": data, "conv_id": conv_id, "created_at": iso_date})

            # Check if the group exists, if not create it
            groups = client.group.get_all_groups()
            groups = dict(groups)["groups"]
            exist_ids = [gp.group_id for gp in groups]
            if conv_id not in exist_ids:
                client.group.add(group_id=conv_id)

            # Add the message to the group
            client.graph.add(
                data=data,
                type="message",
                created_at=iso_date,
                group_id=conv_id,
            )

    elif frame == "memos":
        for chat in tqdm(session, desc=f"{metadata['session_key']}"):
            data = chat.get("speaker") + ": " + chat.get("text")
            print({"context": data, "conv_id": conv_id, "created_at": iso_date})
            msg = [{"role": "user", "content": data}]

            try:
                memories = client.extract(msg)
            except Exception as ex:
                print(f"Error extracting message {msg}: {ex}")
                memories = []
            print(memories)
            client.add(memories)

    elif frame == "memos_mos":
        messages = []
        messages_reverse = []

        for chat in tqdm(session, desc=f"{metadata['session_key']}"):
            data = chat.get("speaker") + ": " + chat.get("text")

            if chat.get("speaker") == metadata["speaker_a"]:
                messages.append({"role": "user", "content": data, "chat_time": iso_date})
                messages_reverse.append(
                    {"role": "assistant", "content": data, "chat_time": iso_date}
                )
            elif chat.get("speaker") == metadata["speaker_b"]:
                messages.append({"role": "assistant", "content": data, "chat_time": iso_date})
                messages_reverse.append({"role": "user", "content": data, "chat_time": iso_date})
            else:
                raise ValueError(
                    f"Unknown speaker {chat.get('speaker')} in session {metadata['session_key']}"
                )

            print({"context": data, "conv_id": conv_id, "created_at": iso_date})

        speaker_a_user_id = conv_id + "_speaker_a"
        speaker_b_user_id = conv_id + "_speaker_b"

        client.add(
            messages=messages,
            user_id=speaker_a_user_id,
        )

        revised_client.add(
            messages=messages_reverse,
            user_id=speaker_b_user_id,
        )
        print(f"Added messages for {speaker_a_user_id} and {speaker_b_user_id} successfully.")

    elif frame == "mem0" or frame == "mem0_graph":
        print(f"Processing abc for {metadata['session_key']}")
        messages = []
        messages_reverse = []

        for chat in tqdm(session, desc=f"{metadata['session_key']}"):
            data = chat.get("speaker") + ": " + chat.get("text")

            if chat.get("speaker") == metadata["speaker_a"]:
                messages.append({"role": "user", "content": data})
                messages_reverse.append({"role": "assistant", "content": data})
            elif chat.get("speaker") == metadata["speaker_b"]:
                messages.append({"role": "assistant", "content": data})
                messages_reverse.append({"role": "user", "content": data})
            else:
                raise ValueError(
                    f"Unknown speaker {chat.get('speaker')} in session {metadata['session_key']}"
                )

            print({"context": data, "conv_id": conv_id, "created_at": iso_date})

        for i in range(0, len(messages), 2):
            batch_messages = messages[i : i + 2]
            batch_messages_reverse = messages_reverse[i : i + 2]

            if frame == "mem0":
                client.add(
                    messages=batch_messages,
                    timestamp=timestamp,
                    user_id=metadata["speaker_a_user_id"],
                    version="v2",
                )
                client.add(
                    messages=batch_messages_reverse,
                    timestamp=timestamp,
                    user_id=metadata["speaker_b_user_id"],
                    version="v2",
                )

            elif frame == "mem0_graph":
                client.add(
                    messages=batch_messages,
                    timestamp=timestamp,
                    user_id=metadata["speaker_a_user_id"],
                    output_format="v1.1",
                    version="v2",
                    enable_graph=True,
                )
                client.add(
                    messages=batch_messages_reverse,
                    timestamp=timestamp,
                    user_id=metadata["speaker_b_user_id"],
                    output_format="v1.1",
                    version="v2",
                    enable_graph=True,
                )

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)

    return elapsed_time


def process_user(conv_idx, frame, locomo_df, version, num_workers=1):
    try:
        conversation = locomo_df["conversation"].iloc[conv_idx]
        max_session_count = 35
        start_time = time.time()
        total_session_time = 0
        valid_sessions = 0

        revised_client = None
        if frame == "zep":
            client = get_client("zep")
        elif frame == "mem0" or frame == "mem0_graph":
            client = get_client(frame)
            client.delete_all(user_id=f"locomo_exp_user_{conv_idx}")
            client.delete_all(user_id=f"{conversation.get('speaker_a')}_{conv_idx}")
            client.delete_all(user_id=f"{conversation.get('speaker_b')}_{conv_idx}")
        elif frame == "memos":
            conv_id = "locomo_exp_user_" + str(conv_idx)
            client = get_client("memos", conv_id, version)
        elif frame == "memos_mos":
            conv_id = "locomo_exp_user_" + str(conv_idx)
            speaker_a_user_id = conv_id + "_speaker_a"
            speaker_b_user_id = conv_id + "_speaker_b"
            client = get_client("memos_mos", speaker_a_user_id, version)
            revised_client = get_client("memos_mos", speaker_b_user_id, version)

        sessions_to_process = []
        for session_idx in range(max_session_count):
            session_key = f"session_{session_idx}"
            session = conversation.get(session_key)
            if session is None:
                continue

            metadata = {
                "session_date": conversation.get(f"session_{session_idx}_date_time") + " UTC",
                "speaker_a": conversation.get("speaker_a"),
                "speaker_b": conversation.get("speaker_b"),
                "speaker_a_user_id": f"{conversation.get('speaker_a')}_{conv_idx}",
                "speaker_b_user_id": f"{conversation.get('speaker_b')}_{conv_idx}",
                "conv_idx": conv_idx,
                "session_key": session_key,
            }
            sessions_to_process.append((session, metadata))
            valid_sessions += 1

        print(
            f"Processing {valid_sessions} sessions for user {conv_idx} with {num_workers} workers"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    ingest_session, client, session, frame, metadata, revised_client
                ): metadata["session_key"]
                for session, metadata in sessions_to_process
            }

            for future in concurrent.futures.as_completed(futures):
                session_key = futures[future]
                try:
                    session_time = future.result()
                    total_session_time += session_time
                    print(f"User {conv_idx}, {session_key} processed in {session_time} seconds")
                except Exception as e:
                    print(f"Error processing user {conv_idx}, session {session_key}: {e!s}")

        if frame == "memos":
            conv_id = "locomo_exp_user_" + str(conv_idx)
            client.dump(f"results/locomo/memos-{version}/storages/{conv_id}")
            del client

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        print(f"User {conv_idx} processed successfully in {elapsed_time} seconds")

        return elapsed_time

    except Exception as e:
        return f"Error processing user {conv_idx}: {e!s}"


async def main_nemori(version="default"):
    """Main function for Nemori processing."""
    load_dotenv()
    locomo_df = pd.read_json("data/locomo/locomo10.json")

    print("🚀 Starting Nemori Ingestion")
    print("=" * 50)

    # Create Nemori experiment
    experiment = NemoriExperiment(version=version, episode_mode="speaker")

    try:
        # Step 1: Setup LLM provider
        llm_available = await experiment.setup_llm_provider()
        if not llm_available:
            print("⚠️ Continuing with fallback mode (no LLM)")

        # Step 2: Load data
        experiment.load_locomo_data(locomo_df)

        # Step 3: Setup storage and retrieval
        await experiment.setup_storage_and_retrieval()

        # Step 4: Build episodes
        await experiment.build_episodes()

        print("\n🎉 Nemori Ingestion Complete")
        print(f"✅ Successfully processed {len(experiment.conversations)} conversations")
        print(f"✅ Created {len(experiment.episodes)} episodes")

    except Exception as e:
        print(f"❌ Nemori ingestion failed: {e}")
        raise
    finally:
        await experiment.cleanup()


def main(frame, version="default", num_workers=4):
    load_dotenv()

    if frame == "nemori":
        if not NEMORI_AVAILABLE:
            print("❌ Nemori is not available. Please install nemori to use this framework.")
            return
        # Run async main for nemori
        asyncio.run(main_nemori(version))
        return

    locomo_df = pd.read_json("data/locomo/locomo10.json")

    num_users = 10
    start_time = time.time()
    total_time = 0

    print(
        f"Starting processing for {num_users} users in serial mode, each user using {num_workers} workers for sessions..."
    )

    for user_id in range(num_users):
        try:
            result = process_user(user_id, frame, locomo_df, version, num_workers)
            if isinstance(result, float):
                total_time += result
            else:
                print(result)
        except Exception as e:
            print(f"Error processing user {user_id}: {e!s}")

    if num_users > 0:
        average_time = total_time / num_users
        minutes = int(average_time // 60)
        seconds = int(average_time % 60)
        average_time_formatted = f"{minutes} minutes and {seconds} seconds"
        print(
            f"The frame {frame} processed {num_users} users in average of {average_time_formatted} per user."
        )

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    elapsed_time = f"{minutes} minutes and {seconds} seconds"
    print(f"Total processing time: {elapsed_time}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        choices=["zep", "memos", "mem0", "mem0_graph", "memos_mos", "nemori"],
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
    args = parser.parse_args()
    lib = args.lib
    version = args.version
    workers = args.workers

    main(lib, version, workers)
