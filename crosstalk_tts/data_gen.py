#!/usr/bin/env python3
"""
使用 macOS 的 say 命令生成英语对话音频和训练数据
"""

import subprocess
import json
from pathlib import Path
import re

# 英语对话内容
DIALOGUE = """
A: cheerful | Hey Karen, have you ever heard of the "veil of ignorance"?
B: curious | The veil of ignorance? Sounds like a fancy mosquito net you sleep under at night.
A: laughing | Haha! You really have an imagination. Not a mosquito net—it's a philosopher's idea. An American philosopher named John Rawls came up with it in "A Theory of Justice."
B: confused | Oh, "A Theory of Justice"? That already sounds way too intellectual. Can my little brain handle it?
A: patient | Of course! Imagine this: before setting rules for a brand-new society, you have to wear the "veil of ignorance."
B: puzzled | A veil? What is this, plastic surgery?
A: explaining | Not surgery—it makes you forget who you are! Gender, race, wealth, talents, family background, beliefs… all wiped clean!
B: surprised | Oh wow, I even forget whether I'm a man or a woman? Then how do I know what clothes to wear?
A: laughing | Haha! Clothes are up to you. What you forget is your social identity. The point is, you don't know if you'll be rich or poor, elite or ordinary, healthy or sick.
B: thoughtful | Ah, so if I set the rules, I'd better be careful—not biased toward anyone, right?
A: agreeing | Exactly! Since you don't know which side you'll end up on, you must create fair rules.
B: determined | Then I'd better look out for the weakest people. Who knows? Tomorrow I could be the weakest myself!
A: approving | That's right! That's Rawls's idea: just rules should also benefit society's most disadvantaged.
B: impressed | Sounds like this "veil of ignorance" is really magical—not for building a utopia, but for helping us step outside ourselves and see the bigger picture.
A: warm | Exactly! With it on, you can calmly think about how to make society fairer so everyone feels secure.
B: joking | Hey Samantha, if I put on this veil, can I avoid doing housework?
A: laughing | Haha! Nope—housework has to be fair too. No one gets a free pass!
B: resigned | Then I'd better wear the veil to think up the rules, but still go home and wash the dishes.
A: cheerful | Right! Rules fair, housework fair—that's real justice!
"""


def classify_emotion(text, emotion_tag):
    """
    根据情感标签和文本内容分类情感
    0: neutral (中性、平静)
    1: excited (兴奋、激动)
    2: sarcastic (讽刺、无奈)
    3: exaggerated (夸张、大笑)
    4: questioning (疑问、好奇)
    """
    # 情感标签映射
    emotion_map = {
        "cheerful": 1,  # 愉快 -> 兴奋
        "curious": 4,  # 好奇 -> 疑问
        "laughing": 3,  # 大笑 -> 夸张
        "confused": 4,  # 困惑 -> 疑问
        "patient": 0,  # 耐心 -> 平静
        "puzzled": 4,  # 困惑 -> 疑问
        "explaining": 0,  # 解释 -> 平静
        "surprised": 3,  # 惊讶 -> 夸张
        "thoughtful": 0,  # 思考 -> 平静
        "agreeing": 1,  # 同意 -> 兴奋
        "determined": 1,  # 坚定 -> 兴奋
        "approving": 1,  # 赞同 -> 兴奋
        "impressed": 1,  # 印象深刻 -> 兴奋
        "warm": 0,  # 温暖 -> 平静
        "joking": 2,  # 开玩笑 -> 讽刺
        "resigned": 2,  # 无奈 -> 讽刺
    }

    # 优先使用标签
    if emotion_tag and emotion_tag.lower() in emotion_map:
        return emotion_map[emotion_tag.lower()]

    # 根据标点符号判断
    if "?" in text:
        return 4  # 疑问
    elif "Haha" in text or "!" in text:
        return 3  # 夸张/大笑
    elif "..." in text or "Ah" in text.lower():
        return 2  # 讽刺/无奈
    else:
        return 0  # 平静


def parse_dialogue(dialogue_text):
    """解析对话文本（格式：A: emotion | text）"""
    lines = [line.strip() for line in dialogue_text.strip().split("\n") if line.strip()]

    dialogue_data = []
    for line in lines:
        # 匹配格式：A: emotion | text
        match = re.match(r"([AB]):\s*(\w+)\s*\|\s*(.+)", line)
        if match:
            speaker_id, emotion_tag, text = match.groups()
            dialogue_data.append(
                {
                    "speaker_id": speaker_id,
                    "emotion_tag": emotion_tag,
                    "text": text.strip(),
                }
            )

    return dialogue_data


def generate_audio_with_say(text, output_path, voice="Samantha", rate=180):
    """
    使用 macOS say 命令生成英语音频并保存为 WAV 格式

    Args:
        text: 要合成的文本
        output_path: 输出音频路径（.wav）
        voice: 语音名称，英文可选：
               - Samantha (女声，美式英语，自然)
               - Alex (男声，美式英语)
               - Karen (女声，澳大利亚英语)
               - Daniel (男声，英式英语)
               - Fiona (女声，苏格兰英语)
        rate: 语速 (默认180，范围约100-300)
    """
    # 创建输出目录
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 方法1: 直接生成 WAV (推荐)
    cmd = [
        "say",
        "-v",
        voice,
        "-r",
        str(rate),
        "-o",
        str(output_path),
        "--data-format=LEI16@24000",  # 16位 PCM, 24kHz 采样率
        text,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Generated audio: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate: {output_path}")
        print(f"  Error: {e.stderr.decode()}")

        # 如果直接生成 WAV 失败，尝试方法2: AIFF + afconvert
        print("  Trying afconvert...")
        return generate_via_afconvert(text, output_path, voice, rate)


def generate_via_afconvert(text, output_path, voice, rate):
    """
    使用 say 生成 AIFF，然后用 afconvert 转为 WAV
    """
    # 生成临时 AIFF 文件
    temp_aiff = output_path.with_suffix(".aiff")

    # 生成 AIFF
    cmd_say = ["say", "-v", voice, "-r", str(rate), "-o", str(temp_aiff), text]

    try:
        subprocess.run(cmd_say, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ AIFF generation failed: {e.stderr.decode()}")
        return False

    # 转换为 WAV
    cmd_convert = [
        "afconvert",
        "-f",
        "WAVE",
        "-d",
        "LEI16@24000",  # 16位整数，24kHz
        str(temp_aiff),
        str(output_path),
    ]

    try:
        subprocess.run(cmd_convert, check=True, capture_output=True)
        # 删除临时文件
        temp_aiff.unlink()
        print(f"✓ Generated audio: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ WAV conversion failed: {e.stderr.decode()}")
        # 保留 AIFF 文件
        return False


def main():
    # 配置
    BASE_DIR = Path("data/raw")
    OUTPUT_JSON = "data/training_data.json"

    # 说话人配置（英语）
    SPEAKERS = {
        "A": {
            "id": "speaker_a",  # 第一说话人（类似捧哏）
            "speaker_id": 0,
            "name": "Samantha",
            "voice": "Samantha",  # 美式英语女声
            "rate": 175,  # 稍慢，显得稳重
        },
        "B": {
            "id": "speaker_b",  # 第二说话人（类似逗哏）
            "speaker_id": 1,
            "name": "Karen",
            "voice": "Karen",  # 澳大利亚英语女声（音色不同）
            "rate": 190,  # 稍快，显得活泼
        },
    }

    # 检查 say 命令是否可用
    try:
        subprocess.run(["say", "-v", "?"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'say' command not found. Please ensure you're on macOS.")
        return

    # 解析对话
    dialogue_data = parse_dialogue(DIALOGUE)

    if not dialogue_data:
        print("Error: No dialogue data parsed. Please check the DIALOGUE format.")
        return

    # 生成样本列表
    samples = []

    # 统计说话人的句子数量
    speaker_counts = {"A": 0, "B": 0}

    for item in dialogue_data:
        speaker_id = item["speaker_id"]
        text = item["text"]
        emotion_tag = item["emotion_tag"]

        if speaker_id not in SPEAKERS:
            print(f"Warning: Unknown speaker {speaker_id}, skipping")
            continue

        speaker_config = SPEAKERS[speaker_id]
        speaker_counts[speaker_id] += 1
        count = speaker_counts[speaker_id]

        # 生成样本ID
        sample_id = f"{speaker_config['id']}_{count:03d}"

        # 音频路径 - 直接使用 .wav
        audio_dir = BASE_DIR / speaker_config["id"]
        audio_filename = f"{count:03d}.wav"
        audio_path = audio_dir / audio_filename

        # 分类情感
        emotion = classify_emotion(text, emotion_tag)

        # 生成音频
        success = generate_audio_with_say(
            text=text,
            output_path=str(audio_path),
            voice=speaker_config["voice"],
            rate=speaker_config["rate"],
        )

        if not success:
            print(f"  Warning: Audio generation failed, skipping this sample")
            continue

        # 添加到样本列表
        samples.append(
            {
                "id": sample_id,
                "audio_path": str(audio_path.relative_to(Path("."))),
                "text": text,
                "speaker": speaker_config["speaker_id"],
                "emotion": emotion,
                "speaker_name": speaker_config["name"],
                "emotion_tag": emotion_tag,
            }
        )

        print(f"  Text: {text[:50]}...")
        print(f"  Emotion: {emotion_tag} -> {emotion}")
        print()

    # 保存 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Complete!")
    print(f"  Generated audio: {len(samples)} samples")
    print(f"  Speaker A (Samantha): {speaker_counts['A']} samples")
    print(f"  Speaker B (Karen): {speaker_counts['B']} samples")
    print(f"  Training data saved to: {OUTPUT_JSON}")
    print(f"{'='*60}")

    # 打印统计信息
    print("\nEmotion Distribution:")
    emotion_names = {
        0: "Neutral/Calm",
        1: "Excited/Happy",
        2: "Sarcastic/Resigned",
        3: "Exaggerated/Laughing",
        4: "Questioning/Curious",
    }
    emotion_counts = {}
    for sample in samples:
        emotion = sample["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    for emotion_id, count in sorted(emotion_counts.items()):
        print(f"  {emotion_names[emotion_id]}: {count} samples")

    # 显示可用的英文语音
    print("\nAvailable English voices on macOS:")
    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
        english_voices = [
            line
            for line in result.stdout.split("\n")
            if "en_" in line.lower() or "english" in line.lower()
        ]
        for voice in english_voices[:10]:  # 显示前10个
            print(f"  {voice}")
    except:
        pass

    print("\nRecommended voices for different styles:")
    print("  - Samantha: Natural female US English (recommended for Speaker A)")
    print("  - Karen: Australian English female (recommended for Speaker B)")
    print("  - Alex: Male US English")
    print("  - Daniel: Male British English")
    print("  - Fiona: Female Scottish English")


if __name__ == "__main__":
    main()
