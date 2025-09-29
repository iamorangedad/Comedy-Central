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

def test_single_speaker_tts():
    """测试单说话人TTS"""
    print("🎤 测试单说话人TTS...")
    
    client = genai.Client(api_key=API_KEY)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents="Say cheerfully: 你好！欢迎使用AI语音生成功能。今天天气真不错！",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='Kore',  # 使用Kore声音
                        )
                    )
                )
            )
        )
        
        # 获取音频数据
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        
        # 保存为WAV文件
        setup_wave_file("single_speaker_test.wav", audio_data)
        print("✅ 单说话人TTS测试完成")
        
    except Exception as e:
        print(f"❌ 单说话人TTS测试失败: {e}")