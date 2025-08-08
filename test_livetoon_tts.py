#!/usr/bin/env python3
"""
LiveToon TTS テストスクリプト
"""

import asyncio
import sys
import os
from pathlib import Path

# プロジェクトルートからのインポートを可能にする
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipecat.services.livetoon.tts import LivetoonTTSService
from pipecat.frames.frames import TTSAudioRawFrame, TTSTextFrame
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio import TaskManager


async def test_livetoon_tts():
    """LiveToon TTSサービスをテストする"""
    print("🎙️ LiveToon TTS テストを開始します...")
    print("-" * 50)
    
    # TaskManagerを初期化
    task_manager = TaskManager()
    
    # TTSサービスを初期化
    tts_service = LivetoonTTSService(
        api_url="https://livetoon-tts.dev-livetoon.com",
        voice_id="default",
        sample_rate=24000,
        language=Language.JA
    )
    
    # TaskManagerを設定
    task_manager.set_event_loop(asyncio.get_event_loop())
    tts_service._task_manager = task_manager
    
    # サービス開始
    from pipecat.frames.frames import StartFrame
    await tts_service.start(StartFrame())
    
    # テストするテキスト
    test_texts = [
        "こんにちは、LiveToon TTSのテストです。",
        "今日はいい天気ですね。",
        "音声合成が正しく動作しているか確認しています。"
    ]
    
    print(f"✅ TTSサービス初期化完了")
    print(f"   - API URL: {tts_service._api_url}")
    print(f"   - Voice ID: {tts_service.voice_id}")
    print(f"   - Sample Rate: {tts_service._sample_rate} Hz")
    print("-" * 50)
    
    # 各テキストでTTSを実行
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 テスト {i}/{len(test_texts)}: \"{text}\"")
        print("   音声生成中...")
        
        try:
            frame_count = 0
            audio_bytes_total = 0
            
            # TTSを実行してフレームを取得
            async for frame in tts_service.run_tts(text):
                if isinstance(frame, TTSAudioRawFrame):
                    frame_count += 1
                    audio_bytes_total += len(frame.audio)
                    if frame_count == 1:
                        print(f"   ✅ 最初の音声フレーム受信 (サイズ: {len(frame.audio)} bytes)")
            
            if frame_count > 0:
                print(f"   ✅ 音声生成完了!")
                print(f"      - フレーム数: {frame_count}")
                print(f"      - 総バイト数: {audio_bytes_total:,} bytes")
                print(f"      - 推定時間: {audio_bytes_total / (24000 * 2):.2f} 秒")
            else:
                print(f"   ⚠️ 音声フレームが生成されませんでした")
                
        except Exception as e:
            print(f"   ❌ エラー発生: {e}")
    
    # サービス停止
    from pipecat.frames.frames import EndFrame
    await tts_service.stop(EndFrame())
    
    print("\n" + "=" * 50)
    print("🎉 LiveToon TTSテスト完了!")


if __name__ == "__main__":
    print("=" * 50)
    print("LiveToon TTS Service Test")
    print("=" * 50)
    
    try:
        asyncio.run(test_livetoon_tts())
    except KeyboardInterrupt:
        print("\n\n⚠️ テストが中断されました")
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()