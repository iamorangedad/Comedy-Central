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

def test_language_detection():
    """测试多语言支持"""
    print("🌍 测试多语言支持...")
    
    client = genai.Client(api_key=API_KEY)
    
    # 测试不同语言的文本
    language_tests = [
        ("中文", "你好，世界！今天天气真不错。"),
        ("English", "Hello, world! The weather is beautiful today."),
        ("日本語", "こんにちは、世界！今日は天気がとても良いです。"),
        ("한국어", "안녕하세요, 세계! 오늘 날씨가 정말 좋습니다."),
        ("Español", "¡Hola, mundo! El clima está hermoso hoy."),
        ("Français", "Bonjour, le monde! Le temps est magnifique aujourd'hui.")
    ]
    
    for lang_name, text in language_tests:
        try:
            print(f"  测试语言: {lang_name}")
            
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name='Kore',
                            )
                        )
                    )
                )
            )
            
            # 获取音频数据
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            
            # 保存为WAV文件
            filename = f"language_test_{lang_name.lower().replace(' ', '_')}.wav"
            setup_wave_file(filename, audio_data)
            
        except Exception as e:
            print(f"  ❌ 语言 {lang_name} 测试失败: {e}")
    
    print("✅ 多语言测试完成")