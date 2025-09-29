#!/usr/bin/env python3
"""
Gemini TTS 对话生成测试用例
基于 https://ai.google.dev/gemini-api/docs/speech-generation
"""

import os
import wave
import base64
from google import genai
from google.genai import types

# 配置API密钥
API_KEY = "your_api_key_here"  # 请替换为你的实际API密钥

def setup_wave_file(filename, pcm_data, channels=1, rate=24000, sample_width=2):
    """保存PCM数据为WAV文件"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    print(f"音频文件已保存: {filename}")


def test_voice_variations():
    """测试不同语音选项"""
    print("🎵 测试不同语音选项...")
    
    client = genai.Client(api_key=API_KEY)
    
    # 测试不同的语音
    voices_to_test = [
        ('Kore', 'Firm'),
        ('Puck', 'Upbeat'), 
        ('Zephyr', 'Bright'),
        ('Fenrir', 'Excitable'),
        ('Leda', 'Youthful')
    ]
    
    test_text = "你好，我是AI语音助手，很高兴为您服务！"
    
    for voice_name, voice_style in voices_to_test:
        try:
            print(f"  测试语音: {voice_name} ({voice_style})")
            
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=test_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    )
                )
            )
            
            # 获取音频数据
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            
            # 保存为WAV文件
            filename = f"voice_test_{voice_name.lower()}.wav"
            setup_wave_file(filename, audio_data)
            
        except Exception as e:
            print(f"  ❌ 语音 {voice_name} 测试失败: {e}")
    
    print("✅ 语音选项测试完成")

