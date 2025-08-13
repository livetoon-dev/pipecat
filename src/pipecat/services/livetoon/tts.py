#
# Copyright (c) 2024–2025, Livetoon Corporation
#
# SPDX-License-Identifier: MIT
#

"""Livetoon text-to-speech service implementations.

This module provides HTTP-based TTS service using Livetoon TTS API
with support for streaming audio and voice customization.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

# Check for required dependencies
try:
    import aiohttp
    import numpy as np
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Livetoon TTS, you need to `pip install pipecat-ai[livetoon]`.")
    raise Exception(f"Missing module: {e}")


class LivetoonTTSParams(BaseModel):
    """Parameters for Livetoon TTS synthesis."""

    voice: str = Field(
        default="default", description="Voice ID (default, men, yasaike, zange, uranai)"
    )
    alpha: float = Field(default=0.3, ge=0.0, le=1.0, description="Voice style control")
    beta: float = Field(default=0.7, ge=0.0, le=1.0, description="Voice emotion control")
    speed: float = Field(default=1.0, gt=0.1, le=4.0, description="Speech speed multiplier")
    language: str = Field(default="ja", description="Language code")


class LivetoonTTSService(TTSService):
    """Livetoon Text-to-Speech service.

    Provides high-quality Japanese speech synthesis using Livetoon TTS API
    with real-time streaming capabilities.
    """

    class InputParams(BaseModel):
        """Input parameters for Livetoon TTS configuration.

        Parameters:
            voice: Voice ID to use for synthesis.
            alpha: Voice style control (0.0 to 1.0).
            beta: Voice emotion control (0.0 to 1.0).
            speed: Speech speed multiplier (0.1 to 4.0).
            language: Language code for synthesis.
        """

        voice: str = "default"
        alpha: float = 0.3
        beta: float = 0.7
        speed: float = 1.0
        language: str = "ja"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str = "https://livetoon-tts.dev-livetoon.com",
        voice_id: str = "default",
        sample_rate: int = 24000,
        language: Language | None = Language.JA,
        params: InputParams | None = None,
        use_ssl: bool = True,
        **kwargs,
    ):
        """Initialize the Livetoon TTS service.

        Args:
            api_key (str | None, optional): API key for authentication. Defaults to None.
            api_url (str, optional): Server URL for Livetoon TTS service.
                Defaults to "https://livetoon-tts.dev-livetoon.com".
            voice_id (str, optional): Voice identifier. Defaults to "default".
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 24000.
            language (Language | None, optional): Language for synthesis. Defaults to Language.JA.
            params (InputParams | None, optional): TTS synthesis parameters. Defaults to None.
            use_ssl (bool, optional): Whether to use SSL for connection. Defaults to True.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(
            sample_rate=sample_rate,
            push_text_frames=False,
            push_stop_frames=True,
            **kwargs,
        )

        self._api_key = api_key
        self._api_url = api_url.rstrip("/")
        self._voice_id = voice_id
        self._sample_rate = sample_rate
        self._language_code = language
        self._use_ssl = use_ssl

        # Set up parameters
        if params is None:
            params = LivetoonTTSService.InputParams(voice=voice_id)
        else:
            params.voice = voice_id

        self._params = params
        self.set_voice(voice_id)

        # HTTP session for API calls
        self._session: aiohttp.ClientSession | None = None

        # Setup resampler if target sample rate is different from source (24kHz)
        self._source_sample_rate = 24000
        self._resampler = None
        if self._sample_rate != self._source_sample_rate:
            self._resampler = create_default_resampler()

        logger.info(
            f"Initialized Livetoon TTS Service - URL: {self._api_url}, Voice: {voice_id}, Sample Rate: {sample_rate}"
        )

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            bool: True as this service supports metric generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the TTS service and initialize HTTP session.

        Args:
            frame: The StartFrame that triggered the start.
        """
        # TaskManagerが初期化されていない場合の警告とガイダンス
        if not hasattr(self, '_task_manager') or self._task_manager is None:
            logger.warning("TaskManager is not initialized. For standalone usage, please initialize TaskManager:")
            logger.warning("from pipecat.utils.asyncio import TaskManager")
            logger.warning("task_manager = TaskManager()")
            logger.warning("task_manager.set_event_loop(asyncio.get_event_loop())")
            logger.warning("tts_service._task_manager = task_manager")
            # 緊急時のフォールバック（推奨されない）
            from pipecat.utils.asyncio import TaskManager
            import asyncio
            self._task_manager = TaskManager()
            self._task_manager.set_event_loop(asyncio.get_event_loop())
            logger.warning("TaskManager auto-initialized as fallback (not recommended for production)")
            
        await super().start(frame)

        # Create persistent session for better performance
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        connector = None
        if self._use_ssl:
            connector = aiohttp.TCPConnector(ssl=True)

        self._session = aiohttp.ClientSession(
            headers=headers, 
            connector=connector, 
            timeout=aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
        )
        logger.debug("Livetoon TTS session started")

    async def stop(self, frame: EndFrame):
        """Stop the TTS service and cleanup resources.

        Args:
            frame: The EndFrame that triggered the stop.
        """
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None
        logger.debug("Livetoon TTS session stopped")

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert Pipecat language to service-specific language code.

        Args:
            language: Pipecat language enum.

        Returns:
            str | None: Service-specific language code or None.
        """
        if hasattr(language, "value"):
            lang_code = language.value.lower()
        else:
            lang_code = str(language).lower()

        # Support various Japanese language codes
        if lang_code in ["ja", "jp", "japanese", "jpn"]:
            return "ja"
        return None

    @property
    def voice_id(self) -> str:
        """Get current voice ID.

        Returns:
            str: Current voice identifier.
        """
        return self._params.voice

    def set_voice(self, voice: str):
        """Set the voice for synthesis.

        Args:
            voice: Voice identifier to set.
        """
        logger.debug(f"Setting Livetoon TTS voice to: {voice}")
        self._params.voice = voice
        self._voice_id = voice

    def update_settings(self, settings: dict[str, Any]):
        """Update TTS settings.

        Args:
            settings: Dictionary of settings to update.
        """
        if "voice" in settings:
            self.set_voice(settings["voice"])
        if "alpha" in settings:
            self._params.alpha = float(settings["alpha"])
        if "beta" in settings:
            self._params.beta = float(settings["beta"])
        if "speed" in settings:
            self._params.speed = float(settings["speed"])

        logger.debug(f"Updated TTS settings: {settings}")

    def _wav_to_audio_array(self, wav_data: bytes) -> np.ndarray:
        """Convert WAV PCM data to numpy array.

        Args:
            wav_data: Raw PCM audio data (without WAV header)

        Returns:
            np.ndarray: Audio array in float32 format [-1, 1]
        """
        try:
            # Check if buffer size is a multiple of 2 (16-bit = 2 bytes)
            if len(wav_data) % 2 != 0:
                logger.warning(
                    f"WAV data size {len(wav_data)} is not a multiple of 2, truncating last byte"
                )
                # Truncate to the nearest even number of bytes
                wav_data = wav_data[:-1]

            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(wav_data, dtype=np.int16)
            # Convert to float32 range [-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0
            return audio_float
        except Exception as e:
            logger.error(f"Error converting WAV data: {e}")
            return np.array([], dtype=np.float32)

    def _split_text_by_punctuation(self, text: str, max_length: int = 100) -> list[str]:
        """Split text by punctuation marks to avoid token limit errors.
        
        Args:
            text: Text to split
            max_length: Maximum length for each chunk (default 100 chars)
            
        Returns:
            List of text chunks
        """
        import re
        
        # 句読点で分割（。！？!?を区切りとする）
        sentences = re.split(r'([。！？!?])', text)
        
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences)):
            sentence = sentences[i]
            
            # 句読点は前の文に含める
            if sentence in '。！？!?':
                if current_chunk:
                    current_chunk += sentence
                    # チャンクが長すぎる場合、または句点で終わった場合は区切る
                    if len(current_chunk) >= max_length or sentence:
                        chunks.append(current_chunk)
                        current_chunk = ""
            else:
                # 新しい文を追加
                if len(current_chunk) + len(sentence) > max_length:
                    # 現在のチャンクを保存して新しいチャンクを開始
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk += sentence
        
        # 残りのテキストを追加
        if current_chunk.strip():
            chunks.append(current_chunk)
        
        # 空のチャンクを除外
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        logger.debug(f"Split text into {len(chunks)} chunks: {[len(c) for c in chunks]} chars")
        return chunks

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio frames from text using Livetoon TTS API.

        Args:
            text: Text to synthesize

        Yields:
            Frame: TTS frames (TTSStartedFrame, TTSTextFrame, TTSAudioRawFrame, TTSStoppedFrame)
        """
        logger.debug(f"Generating Livetoon TTS: [{text}]")

        if not text.strip():
            logger.warning("Empty text provided for TTS")
            return

        try:
            # Initialize session if needed
            if not self._session:
                self._session = aiohttp.ClientSession()

            # Start TTS metrics
            await self.start_ttfb_metrics()

            # Emit TTS started frame
            yield TTSStartedFrame()
            yield TTSTextFrame(text)

            # Split long text by punctuation to avoid token limit errors
            text_chunks = self._split_text_by_punctuation(text)
            
            for chunk_idx, text_chunk in enumerate(text_chunks):
                logger.debug(f"Processing chunk {chunk_idx + 1}/{len(text_chunks)}: [{text_chunk[:50]}...]")
                
                # Prepare request data for Livetoon TTS API
                json_data = {
                    "text": text_chunk,
                    "voicepack": self._params.voice,
                    "alpha": self._params.alpha,
                    "beta": self._params.beta,
                    "speed": self._params.speed,
                }

                # Use regular endpoint and chunk the complete WAV file
                # This is more efficient than pseudo-streaming since server pre-generates the WAV
                regular_url = f"{self._api_url}/speak"

                async with self._session.post(regular_url, json=json_data) as response:
                    if response.status == 200:
                        # Get inference time from headers
                        inference_time = response.headers.get("X-Inference-Time")
                        if inference_time:
                            logger.debug(f"TTS inference time for chunk {chunk_idx + 1}: {inference_time}s")

                        # Stop TTFB metrics only on first chunk
                        if chunk_idx == 0:
                            await self.stop_ttfb_metrics()

                        # Read complete WAV file
                        audio_data = await response.read()
                        logger.debug(f"Received complete WAV file for chunk {chunk_idx + 1}: {len(audio_data)} bytes")

                        # Process complete audio data and chunk it for streaming
                        if audio_data.startswith(b"RIFF"):
                            # Skip WAV header (44 bytes) to get raw PCM data
                            pcm_data = audio_data[44:]
                            logger.debug(f"Extracted {len(pcm_data)} bytes of PCM data from WAV")

                            # Split into optimal chunks for streaming simulation
                            chunk_size = 1024  # 1KB chunks - standard size used by most TTS services
                            total_chunks = (len(pcm_data) + chunk_size - 1) // chunk_size
                            
                            logger.debug(f"Chunking into {total_chunks} chunks of {chunk_size} bytes")
                            
                            for i in range(0, len(pcm_data), chunk_size):
                                chunk = pcm_data[i : i + chunk_size]
                                yield await self._create_audio_frame(chunk)
                                logger.debug(f"Processed chunk {i//chunk_size + 1}/{total_chunks}: {len(chunk)} bytes")

                            logger.debug(f"TTS chunk {chunk_idx + 1} completed - chunked {len(pcm_data)} bytes into {total_chunks} frames")
                        else:
                            logger.error(f"Invalid audio format received for chunk {chunk_idx + 1} (not WAV)")
                            yield ErrorFrame("Invalid audio format from TTS API")
                            return
                    else:
                        error_text = await response.text()
                        logger.error(f"TTS API failed for chunk {chunk_idx + 1}: {response.status} - {error_text}")
                        yield ErrorFrame(f"TTS API failed: {response.status}")
                        return

        except aiohttp.ClientError as e:
            logger.exception(f"HTTP error in Livetoon TTS: {e}")
            yield ErrorFrame(f"HTTP error: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error in Livetoon TTS: {e}")
            yield ErrorFrame(f"TTS error: {str(e)}")
        finally:
            # Emit TTS completed frame
            yield TTSStoppedFrame()

    async def _create_audio_frame(self, pcm_data: bytes) -> TTSAudioRawFrame:
        """Create an audio frame from PCM data with optional resampling.

        Args:
            pcm_data: Raw PCM audio data at source sample rate (24kHz)

        Returns:
            TTSAudioRawFrame: Audio frame at target sample rate
        """
        try:
            # Check if buffer size is valid for int16
            if len(pcm_data) % 2 != 0:
                logger.warning(f"PCM data size {len(pcm_data)} is not even, truncating")
                pcm_data = pcm_data[:-1]

            # Resample if needed
            if self._resampler:
                # Resample from 24kHz to target sample rate
                resampled_data = await self._resampler.resample(
                    pcm_data, self._source_sample_rate, self._sample_rate
                )
                logger.debug(
                    f"Resampled {len(pcm_data)} bytes @ {self._source_sample_rate}Hz to {len(resampled_data)} bytes @ {self._sample_rate}Hz"
                )
                pcm_data = resampled_data

            # Create frame with appropriate sample rate
            frame = TTSAudioRawFrame(audio=pcm_data, sample_rate=self._sample_rate, num_channels=1)

            logger.debug(f"Created audio frame: {len(pcm_data)} bytes at {self._sample_rate}Hz")
            return frame

        except Exception as e:
            logger.error(f"Error creating audio frame: {e}")
            # Return empty frame in case of error
            return TTSAudioRawFrame(audio=b"", sample_rate=self._sample_rate, num_channels=1)

    @classmethod
    def get_service_config(cls) -> dict[str, Any]:
        """Get service configuration for Livetoon TTS.

        Returns:
            dict: Service configuration dictionary.
        """
        return {
            "service_type": "tts",
            "service_name": "livetoon_tts",
            "service_class": cls.__name__,
            "supported_languages": ["ja", "jp", "japanese"],
            "configuration": {
                "api_url": "https://livetoon-tts.dev-livetoon.com",
                "api_key": None,  # Set your API key here
                "voice_id": "default",
                "sample_rate": 24000,
                "params": {"alpha": 0.3, "beta": 0.7, "speed": 1.0, "language": "ja"},
            },
            "features": [
                "streaming",
                "voice_customization",
                "parameter_control",
                "japanese_synthesis",
                "real_time_processing",
            ],
            "performance": {
                "first_chunk_latency_ms": 260,
                "chunk_size_bytes": 1024,
                "sample_rate_hz": 24000,
                "channels": 1,
                "bit_depth": 16,
            },
        }


# Export for plugin discovery
__all__ = ["LivetoonTTSService", "LivetoonTTSParams"]
