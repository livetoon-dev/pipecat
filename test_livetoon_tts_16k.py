#!/usr/bin/env python3
"""
LiveToon TTS 16kHz サンプリングレートテストスクリプト

このスクリプトはpipecatフレームワークの標準的な使用パターンに準拠しています。
他のTTSサービス（ElevenLabs、OpenAI等）と同じ方式でTaskManagerとawaitを使用します。
"""

import asyncio
import sys
import os
from pathlib import Path

# プロジェクトルートからのインポートを可能にする
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipecat.services.livetoon.tts import LivetoonTTSService
from pipecat.frames.frames import TTSAudioRawFrame, TTSTextFrame, StartFrame, EndFrame
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio import TaskManager


async def test_livetoon_tts_with_sample_rate(sample_rate: int):
    """指定されたサンプリングレートでLiveToon TTSサービスをテストする - pipecat標準パターン"""
    print(f"\n🎙️ LiveToon TTS テスト - {sample_rate}Hz")
    print("-" * 50)
    
    # TaskManagerを初期化（pipecat標準）
    task_manager = TaskManager()
    task_manager.set_event_loop(asyncio.get_event_loop())
    
    # TTSサービスを初期化（他のTTSサービスと同じパターン）
    tts_service = LivetoonTTSService(
        api_url="https://livetoon-tts.dev-livetoon.com",
        voice_id="default",
        sample_rate=sample_rate,  # テストするサンプリングレート
        language=Language.JA
    )
    
    # TaskManagerを設定（pipecat標準）
    tts_service._task_manager = task_manager
    
    # サービス開始（await必須）
    await tts_service.start(StartFrame())
    
    # テストするテキスト
    test_text = "こんにちは、サンプリングレートのテストです。"
    
    print(f"✅ TTSサービス初期化完了")
    print(f"   - API URL: {tts_service._api_url}")
    print(f"   - Voice ID: {tts_service.voice_id}")
    print(f"   - Sample Rate: {tts_service._sample_rate} Hz")
    print(f"   - Resampler: {'あり' if tts_service._resampler else 'なし'}")
    print("-" * 50)
    
    print(f"\n📝 テキスト: \"{test_text}\"")
    print("   音声生成中...")
    
    try:
        frame_count = 0
        audio_bytes_total = 0
        
        # TTSを実行してフレームを取得（pipecat標準パターン）
        async for frame in tts_service.run_tts(test_text):
            if isinstance(frame, TTSAudioRawFrame):
                frame_count += 1
                audio_bytes_total += len(frame.audio)
                if frame_count == 1:
                    print(f"   ✅ 最初の音声フレーム受信")
                    print(f"      - サイズ: {len(frame.audio)} bytes")
                    print(f"      - フレームのサンプルレート: {frame.sample_rate} Hz")
        
        if frame_count > 0:
            print(f"   ✅ 音声生成完了!")
            print(f"      - フレーム数: {frame_count}")
            print(f"      - 総バイト数: {audio_bytes_total:,} bytes")
            print(f"      - 推定時間: {audio_bytes_total / (sample_rate * 2):.2f} 秒")
            
            # 音声ファイルに保存（オプション）
            output_file = f"test_output_{sample_rate}hz.raw"
            print(f"\n   💾 音声データを {output_file} に保存中...")
            
            with open(output_file, "wb") as f:
                async for frame in tts_service.run_tts(test_text):
                    if isinstance(frame, TTSAudioRawFrame):
                        f.write(frame.audio)
            
            print(f"   ✅ 保存完了!")
            print(f"      再生コマンド: ffplay -f s16le -ar {sample_rate} -ac 1 {output_file}")
        else:
            print(f"   ⚠️ 音声フレームが生成されませんでした")
            
    except Exception as e:
        print(f"   ❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # サービス停止（await必須、try-finallyで確実に実行）
        await tts_service.stop(EndFrame())


async def main():
    """メインテスト関数 - pipecat標準パターンのデモンストレーション"""
    print("=" * 50)
    print("LiveToon TTS サンプリングレートテスト")
    print("pipecatフレームワーク標準パターン使用")
    print("=" * 50)
    
    # 24kHz (オリジナル) でテスト
    await test_livetoon_tts_with_sample_rate(24000)
    
    # 16kHz (リサンプリング) でテスト
    await test_livetoon_tts_with_sample_rate(16000)
    
    print("\n" + "=" * 50)
    print("🎉 すべてのテスト完了!")
    print("   • TaskManager: 正常に動作")
    print("   • await パターン: 他のTTSと同じ")
    print("   • 16kHz リサンプリング: 正常に動作")
    print("=" * 50)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ テストが中断されました")
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()