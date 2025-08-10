#!/usr/bin/env python3
"""
LiveToon TTS サービス使用例

インストール方法:
uv add "git+https://github.com/livetoon-dev/pipecat.git@feature/kotoba-asr#egg=pipecat-ai[livetoon]"

重要: pipecatフレームワークでは全てのTTSサービス（ElevenLabs、OpenAI等）で
TaskManagerとawaitが必須です。これはフレームワークの統一された設計です。
"""

import asyncio
from pipecat.services.livetoon.tts import LivetoonTTSService
from pipecat.frames.frames import TTSAudioRawFrame, StartFrame, EndFrame
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio import TaskManager


async def main():
    """LiveToon TTS使用例 - 他のTTSサービスと同じパターン"""
    
    # TaskManagerを初期化（pipecat全体で必須）
    task_manager = TaskManager()
    task_manager.set_event_loop(asyncio.get_event_loop())
    
    # TTSサービスを初期化
    tts_service = LivetoonTTSService(
        api_url="https://livetoon-tts.dev-livetoon.com",
        voice_id="default",
        sample_rate=16000,  # 16kHz推奨（24kHzも可能）
        language=Language.JA
    )
    
    # TaskManagerを設定（pipecat全体で必須）
    tts_service._task_manager = task_manager
    
    try:
        # サービス開始（await必須）
        await tts_service.start(StartFrame())
        
        # 音声合成実行
        text = "こんにちは、LiveToon TTSのテストです。"
        print(f"🎙️ 音声合成中: {text}")
        print(f"   Sample Rate: {tts_service._sample_rate}Hz")
        print(f"   Resampler: {'有効' if tts_service._resampler else '無効'}")
        
        frame_count = 0
        audio_data = b""
        
        async for frame in tts_service.run_tts(text):
            if isinstance(frame, TTSAudioRawFrame):
                frame_count += 1
                audio_data += frame.audio
                if frame_count == 1:
                    print(f"   ✅ 最初のフレーム: {len(frame.audio)} bytes @ {frame.sample_rate}Hz")
        
        print(f"✅ 完了: {frame_count}フレーム, 合計{len(audio_data):,}bytes")
        
        # 音声ファイル保存（オプション）
        output_file = f"output_{tts_service._sample_rate}hz.raw"
        with open(output_file, "wb") as f:
            f.write(audio_data)
        print(f"💾 音声保存: {output_file}")
        print(f"   再生コマンド: ffplay -f s16le -ar {tts_service._sample_rate} -ac 1 {output_file}")
        
    finally:
        # サービス停止（await必須）
        await tts_service.stop(EndFrame())


# シンプルな使用例（上級者向け）
async def simple_usage():
    """最小限のコード例"""
    task_manager = TaskManager()
    task_manager.set_event_loop(asyncio.get_event_loop())
    
    tts = LivetoonTTSService(
        api_url="https://livetoon-tts.dev-livetoon.com",
        sample_rate=16000  # ← これが主要な設定
    )
    tts._task_manager = task_manager
    
    await tts.start(StartFrame())
    
    async for frame in tts.run_tts("テスト"):
        if isinstance(frame, TTSAudioRawFrame):
            # 音声データを処理
            print(f"音声データ: {len(frame.audio)} bytes")
    
    await tts.stop(EndFrame())


if __name__ == "__main__":
    print("=" * 60)
    print("LiveToon TTS Service - 完全な使用例")
    print("=" * 60)
    asyncio.run(main())