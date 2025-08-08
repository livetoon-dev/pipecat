#
# Copyright (c) 2024–2025, Livetoon Corporation
#
# SPDX-License-Identifier: MIT
#

"""LiveToon speech-to-text service implementations.

This module provides HTTP-based STT service using LiveToon Parakeet Japanese STT API
with support for both streaming and batch audio processing.
"""

import asyncio
import io
import tempfile
from typing import Any, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

# Check for required dependencies
try:
    import aiohttp
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use LiveToon STT, you need to `pip install pipecat-ai[livetoon]`.")
    raise Exception(f"Missing module: {e}")


class LivetoonSTTParams(BaseModel):
    """Parameters for LiveToon Parakeet STT service."""

    decoding_type: str = Field(
        default="tdt", description="Decoding method: 'tdt' or 'ctc'"
    )
    sample_rate: int = Field(
        default=16000, description="Audio sample rate in Hz"
    )
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )


class LiveToonSTTService(STTService):
    """LiveToon Parakeet Japanese Speech-to-Text service.

    Provides high-quality Japanese speech recognition using LiveToon Parakeet STT API
    with support for real-time audio processing.
    """

    class InputParams(BaseModel):
        """Input parameters for LiveToon STT configuration.

        Parameters:
            decoding_type: Decoding method to use ('tdt' or 'ctc').
            confidence_threshold: Minimum confidence score for transcriptions.
            vad_enabled: Whether to use Voice Activity Detection for dynamic buffering.
            silence_threshold: Silence duration (seconds) to trigger transcription.
        """

        decoding_type: str = "tdt"
        confidence_threshold: float = 0.5
        vad_enabled: bool = True  # Enable VAD-based dynamic buffering
        silence_threshold: float = 1.0  # 1 second of silence triggers transcription

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str = "https://livetoon-stt.dev-livetoon.com",
        sample_rate: int = 16000,
        language: Language | None = Language.JA,
        params: InputParams | None = None,
        use_ssl: bool = True,
        **kwargs,
    ):
        """Initialize the LiveToon STT service.

        Args:
            api_key (str | None, optional): API key for authentication. Defaults to None.
            api_url (str, optional): Server URL for LiveToon STT service.
                Defaults to "https://livetoon-stt.dev-livetoon.com".
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 16000.
            language (Language | None, optional): Language for recognition. Defaults to Language.JA.
            params (InputParams | None, optional): STT parameters. Defaults to None.
            use_ssl (bool, optional): Whether to use SSL for connection. Defaults to True.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )

        self._api_key = api_key
        self._api_url = api_url.rstrip("/")
        self._sample_rate = sample_rate
        self._language_code = language
        self._use_ssl = use_ssl

        # Set up parameters
        if params is None:
            params = LiveToonSTTService.InputParams()

        self._params = params

        # HTTP session for API calls
        self._session: aiohttp.ClientSession | None = None

        # Audio buffer for VAD-based dynamic buffering
        self._audio_buffer = bytearray()
        self._is_speaking = False
        self._silence_threshold = params.silence_threshold if hasattr(params, 'silence_threshold') else 1.0

        logger.info(
            f"Initialized LiveToon STT Service - URL: {self._api_url}, Sample Rate: {sample_rate}, VAD-based dynamic buffering enabled"
        )

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            bool: True as this service supports metric generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the STT service and initialize HTTP session.

        Args:
            frame: The StartFrame that triggered the start.
        """
        await super().start(frame)

        # Create persistent session for better performance
        headers = {"Content-Type": "multipart/form-data"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        connector = None
        if self._use_ssl:
            connector = aiohttp.TCPConnector(ssl=True)

        self._session = aiohttp.ClientSession(
            connector=connector, timeout=aiohttp.ClientTimeout(total=30)
        )
        logger.debug("LiveToon STT session started")

    async def stop(self, frame: EndFrame):
        """Stop the STT service and cleanup resources.

        Args:
            frame: The EndFrame that triggered the stop.
        """
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None
        logger.debug("LiveToon STT session stopped")

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

    @traced_stt
    async def run_stt(self, audio: bytes) -> None:
        """Process audio chunk using VAD-based dynamic buffering.

        Args:
            audio: Raw audio data bytes
        """
        if not audio:
            return

        # Always accumulate audio when speaking
        if self._is_speaking:
            self._audio_buffer.extend(audio)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames including VAD events."""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("Speech started - beginning audio accumulation")
            self._is_speaking = True
            self._audio_buffer = bytearray()  # Reset buffer for new speech
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("Speech stopped - processing accumulated audio")
            self._is_speaking = False
            if len(self._audio_buffer) > 0:
                await self._process_buffer()

    async def _process_buffer(self):
        """Process accumulated audio buffer and generate transcription."""
        if len(self._audio_buffer) == 0:
            return

        try:
            # Initialize session if needed
            if not self._session:
                await self.start(StartFrame())

            # Create temporary WAV file from buffer
            audio_data = bytes(self._audio_buffer)
            wav_data = self._create_wav_from_pcm(audio_data)

            # Clear buffer
            self._audio_buffer = bytearray()

            # Send to STT API
            transcription_result = await self._transcribe_audio(wav_data)

            if transcription_result and transcription_result.get("text"):
                text = transcription_result["text"].strip()
                confidence = transcription_result.get("confidence", 1.0)

                if text and (confidence is None or confidence >= self._params.confidence_threshold):
                    confidence_str = f"{confidence:.2f}" if confidence is not None else "N/A"
                    logger.debug(f"STT transcription: [{text}] (confidence: {confidence_str})")
                    
                    # Emit transcription frames
                    await self.push_frame(TranscriptionFrame(text, "", time_now_iso8601()))

        except Exception as e:
            logger.exception(f"Error processing STT buffer: {e}")
            await self.push_frame(ErrorFrame(f"STT error: {str(e)}"))

    def _create_wav_from_pcm(self, pcm_data: bytes) -> bytes:
        """Create WAV file header + PCM data.

        Args:
            pcm_data: Raw PCM audio data

        Returns:
            bytes: Complete WAV file data
        """
        import struct

        # WAV header for 16-bit mono PCM
        sample_rate = self._sample_rate
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm_data)
        
        header = struct.pack(
            '<4sL4s4sLHHLLHH4sL',
            b'RIFF',                    # ChunkID
            36 + data_size,             # ChunkSize
            b'WAVE',                    # Format
            b'fmt ',                    # Subchunk1ID
            16,                         # Subchunk1Size (PCM)
            1,                          # AudioFormat (PCM)
            num_channels,               # NumChannels
            sample_rate,                # SampleRate
            byte_rate,                  # ByteRate
            block_align,                # BlockAlign
            bits_per_sample,            # BitsPerSample
            b'data',                    # Subchunk2ID
            data_size                   # Subchunk2Size
        )
        
        return header + pcm_data

    async def _transcribe_audio(self, wav_data: bytes) -> dict[str, Any] | None:
        """Send audio to LiveToon STT API for transcription.

        Args:
            wav_data: Complete WAV file data

        Returns:
            dict: Transcription result with text and confidence
        """
        try:
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('file', io.BytesIO(wav_data), filename='audio.wav', content_type='audio/wav')
            data.add_field('decoding_type', self._params.decoding_type)

            # Send transcription request
            transcribe_url = f"{self._api_url}/transcribe"

            async with self._session.post(transcribe_url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"STT API response: {result}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"STT API error: {response.status} - {error_text}")
                    return None

        except aiohttp.ClientError as e:
            logger.exception(f"HTTP error in LiveToon STT: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error in LiveToon STT: {e}")
            return None

    async def flush_audio_buffer(self):
        """Process any remaining audio in buffer."""
        if len(self._audio_buffer) > 0:
            logger.debug(f"Flushing remaining {len(self._audio_buffer)} bytes from STT buffer")
            await self._process_buffer()

    @classmethod
    def get_service_config(cls) -> dict[str, Any]:
        """Get service configuration for LiveToon STT.

        Returns:
            dict: Service configuration dictionary.
        """
        return {
            "service_type": "stt",
            "service_name": "livetoon_stt",
            "service_class": cls.__name__,
            "supported_languages": ["ja", "jp", "japanese"],
            "configuration": {
                "api_url": "https://livetoon-stt.dev-livetoon.com",
                "api_key": None,  # Set your API key here
                "sample_rate": 16000,
                "params": {
                    "decoding_type": "tdt",
                    "confidence_threshold": 0.5,
                    "vad_enabled": True,
                    "silence_threshold": 1.0,
                },
            },
            "features": [
                "japanese_recognition",
                "high_accuracy",
                "real_time_processing",
                "confidence_scoring",
                "multiple_decoding_methods",
            ],
            "performance": {
                "vad_based": True,
                "sample_rate_hz": 16000,
                "channels": 1,
                "bit_depth": 16,
                "real_time_processing": True,
            },
        }


# Export for plugin discovery
__all__ = ["LiveToonSTTService", "LivetoonSTTParams"]