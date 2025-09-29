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
API_KEY = "todo"  # 请替换为你的实际API密钥


def setup_wave_file(filename, pcm_data, channels=1, rate=24000, sample_width=2):
    """保存PCM数据为WAV文件"""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    print(f"音频文件已保存: {filename}")


def test_multi_speaker_dialogue():
    """测试多说话人对话生成"""
    print("🎭 测试多说话人对话生成...")

    client = genai.Client(api_key=API_KEY)

    # 定义对话内容
    dialogue_prompt = """
    请生成以下对话的音频：
    
    张三：李四啊，你听说过“无知之幕”吗？
    李四：无知之幕？听着像是晚上睡觉的蚊帐？
    张三：哈哈，你可真会联想。不是蚊帐，是哲学家的点子。美国有个哲学家叫罗尔斯，他在《正义论》里提出来的。
    李四：哦，《正义论》？听着就高大上，我这小脑袋能听得懂吗？
    张三：能！你想啊，你要给一个新社会定规则，可在这之前，你得戴上一张“无知之幕”。
    李四：戴幕布？是整容啊？
    张三：不是整容，是让你忘掉自己是谁！性别、种族、财富、天赋、家庭背景、信仰……全忘光！
    李四：哎哟，我连自己是男是女都忘了，那穿衣服怎么办？
    张三：哈哈，穿衣服随便，你忘的是社会身份！重点是，你不知道自己将来会是富人还是穷人，是精英还是普通工人，是健康还是生病。
    李四：哎呀，那我定规则可得小心了，偏心不得啊？
    张三：没错！你不知道自己会站在哪一边，所以必须制定公平的规则。
    李四：那我肯定得照顾最弱势的人呗，说不定明天我就是最弱势的那位！
    张三：正解！这就是罗尔斯的意思：公平的规则应该让社会最弱的人也受益。
    李四：听起来这“无知之幕”可真神奇，不是用来造乌托邦，而是帮我们跳出自我，看得更远。
    张三：没错！戴上它，你就能心平气和地想，怎么让社会更公正，让每个人都安心。
    李四：哎呀张三，那我戴上这个幕布，是不是可以逃避家务啊？
    张三：哈哈，家务也得公平执行，谁都不能例外！
    李四：那我看还是戴着去想规则，回家照样洗碗吧……
    张三：对，规则公平，家务也公平，社会才正义嘛！
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=dialogue_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=[
                            types.SpeakerVoiceConfig(
                                speaker="张三",
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name="Kore",  # 男性声音
                                    )
                                ),
                            ),
                            types.SpeakerVoiceConfig(
                                speaker="李四",
                                voice_config=types.VoiceConfig(
                                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                        voice_name="Puck",  # 女性声音
                                    )
                                ),
                            ),
                        ]
                    )
                ),
            ),
        )

        # 获取音频数据
        audio_data = response.candidates[0].content.parts[0].inline_data.data

        # 保存为WAV文件
        setup_wave_file("multi_speaker_dialogue.wav", audio_data)
        print("✅ 多说话人对话测试完成") // 耗时87.97s

    except Exception as e:
        print(f"❌ 多说话人对话测试失败: {e}")
