#!/usr/bin/env python3
"""Test LiveToon STT service with coefont.mp3 audio file."""

import asyncio
import sys
import os
from pathlib import Path

# Add pipecat src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipecat.services.livetoon.stt import LiveToonSTTService
from pipecat.frames.frames import (
    StartFrame,
    EndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection


async def convert_mp3_to_pcm(mp3_path: str, sample_rate: int = 16000) -> bytes:
    """Convert MP3 file to PCM format suitable for STT."""
    try:
        import subprocess
        import tempfile
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
        
        # Convert MP3 to WAV using ffmpeg
        cmd = [
            'ffmpeg', '-i', mp3_path,
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            '-f', 'wav',  # WAV format
            tmp_wav_path,
            '-y'  # Overwrite
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return b""
        
        # Read WAV file and extract PCM data
        with open(tmp_wav_path, 'rb') as f:
            wav_data = f.read()
            # Skip WAV header (44 bytes) to get raw PCM
            pcm_data = wav_data[44:]
        
        # Clean up temp file
        os.unlink(tmp_wav_path)
        
        return pcm_data
        
    except Exception as e:
        print(f"Error converting MP3: {e}")
        return b""


async def test_stt_with_audio():
    """Test LiveToon STT with audio file."""
    
    # Initialize STT service
    stt = LiveToonSTTService(
        api_url="https://livetoon-stt.dev-livetoon.com",
        sample_rate=16000,
    )
    
    # Start the service
    await stt.start(StartFrame())
    
    # Convert MP3 to PCM
    mp3_path = "your_mp3_file.mp3"
    if not os.path.exists(mp3_path):
        print(f"Error: {mp3_path} not found")
        return
    
    print(f"Converting {mp3_path} to PCM...")
    pcm_data = await convert_mp3_to_pcm(mp3_path)
    
    if not pcm_data:
        print("Failed to convert audio")
        return
    
    print(f"PCM data size: {len(pcm_data)} bytes")
    
    # Simulate VAD events
    print("Simulating speech detection...")
    
    # Send UserStartedSpeaking to trigger buffer accumulation
    await stt.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    
    # Process audio in chunks (simulate streaming)
    chunk_size = 8192
    for i in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[i:i+chunk_size]
        await stt.run_stt(chunk)
    
    # Send UserStoppedSpeaking to trigger transcription
    print("Processing accumulated audio...")
    await stt.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    
    # Wait a bit for processing
    await asyncio.sleep(2)
    
    # Stop the service
    await stt.stop(EndFrame())
    
    print("Test completed")


if __name__ == "__main__":
    # Check if ffmpeg is available
    import subprocess
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except:
        print("Error: ffmpeg is required for MP3 conversion")
        print("Install with: brew install ffmpeg")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_stt_with_audio())