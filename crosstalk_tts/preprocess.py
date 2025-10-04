import librosa
import numpy as np
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import re


class EnglishTTSDataProcessor:
    """英语TTS数据预处理器"""

    def __init__(self, config):
        self.sample_rate = config["sample_rate"]  # 24000
        self.n_fft = config["n_fft"]  # 1024
        self.hop_length = config["hop_length"]  # 256
        self.n_mels = config["n_mels"]  # 80
        self.max_wav_value = 32768.0

    def load_and_resample(self, audio_path):
        """加载并重采样音频"""
        wav, sr = librosa.load(audio_path, sr=None)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)

        # 归一化
        if np.abs(wav).max() > 0:
            wav = wav / np.abs(wav).max() * 0.95
        return wav

    def extract_mel_spectrogram(self, wav):
        """提取Mel频谱图"""
        mel_spec = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=8000,
        )

        # 转换为对数刻度
        mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
        return mel_spec

    def _interpolate_f0(self, f0, voiced_flag):
        """插值填充音高的无声部分"""
        f0 = f0.copy()
        f0[~voiced_flag] = np.nan

        # 线性插值
        nans = np.isnan(f0)
        if not nans.all():
            x = lambda z: z.nonzero()[0]
            f0[nans] = np.interp(x(nans), x(~nans), f0[~nans])
        else:
            f0[:] = 0

        return f0

    def extract_pitch(self, wav):
        """提取音高特征"""
        # 英语音高范围略有不同
        f0, voiced_flag, _ = librosa.pyin(
            wav,
            fmin=60,  # 英语可能有更低的音高
            fmax=500,  # 英语可能有更高的音高
            sr=self.sample_rate,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )

        # 插值填充unvoiced部分
        f0 = self._interpolate_f0(f0, voiced_flag)
        return f0

    def extract_energy(self, wav):
        """提取能量特征"""
        energy = librosa.feature.rms(
            y=wav, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        return energy

    def text_to_phoneme_with_g2p(self, text):
        """
        使用 g2p 库将文本转换为音素（推荐方法）
        需要安装: pip install g2p-en
        """
        from g2p_en import G2p

        g2p = G2p()

        # 文本清理
        text = text.lower()

        # 转换为音素
        phonemes = g2p(text)

        # 添加特殊标记
        phonemes = ["<SOS>"] + phonemes + ["<EOS>"]
        return phonemes

    def process_single_sample(self, sample_info):
        """处理单个样本"""
        try:
            # 加载音频
            wav = self.load_and_resample(sample_info["audio_path"])

            # 提取特征
            mel = self.extract_mel_spectrogram(wav)
            pitch = self.extract_pitch(wav)
            energy = self.extract_energy(wav)

            # 处理文本
            phonemes = self.text_to_phoneme_with_g2p(sample_info["text"])

            # 构建处理后的数据
            processed = {
                "id": sample_info["id"],
                "speaker": sample_info["speaker"],  # 0 or 1
                "emotion": sample_info.get("emotion", 0),  # 0-4
                "text": sample_info["text"],
                "phonemes": phonemes,
                "mel": mel.T,  # [time, n_mels]
                "pitch": pitch,
                "energy": energy,
                "wav": wav,
            }

            return processed

        except Exception as e:
            print(f"Error processing {sample_info['id']}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def build_vocabulary(self, all_phonemes):
        """构建音素词表"""
        vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}

        # 统计所有音素
        unique_phonemes = set()
        for phonemes in all_phonemes:
            unique_phonemes.update(phonemes)

        # 移除特殊标记
        unique_phonemes = unique_phonemes - {"<SOS>", "<EOS>"}

        # 添加到词表 (按字母顺序排序)
        for idx, phoneme in enumerate(sorted(unique_phonemes), start=4):
            vocab[phoneme] = idx

        return vocab

    def save_processed_data(self, processed_samples, output_dir):
        """保存处理后的数据"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存元数据
        metadata = []
        for sample in processed_samples:
            if sample is None:
                continue

            sample_id = sample["id"]

            # 保存numpy数组
            np.save(output_dir / f"{sample_id}_mel.npy", sample["mel"])
            np.save(output_dir / f"{sample_id}_pitch.npy", sample["pitch"])
            np.save(output_dir / f"{sample_id}_energy.npy", sample["energy"])
            np.save(output_dir / f"{sample_id}_wav.npy", sample["wav"])

            # 添加到元数据
            metadata.append(
                {
                    "id": sample_id,
                    "speaker": sample["speaker"],
                    "emotion": sample["emotion"],
                    "text": sample["text"],
                    "phonemes": sample["phonemes"],
                }
            )

        # 保存元数据
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # 构建并保存词表
        all_phonemes = [item["phonemes"] for item in metadata]
        vocab = self.build_vocabulary(all_phonemes)

        with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        print(f"Processed {len(metadata)} samples")
        print(f"Vocabulary size: {len(vocab)}")

        # 显示词表样例
        print("\nVocabulary sample (first 20):")
        vocab_items = list(vocab.items())[:20]
        for phoneme, idx in vocab_items:
            print(f"  '{phoneme}': {idx}")


def load_training_data(json_path):
    """从生成的 training_data.json 加载样本"""
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return samples


# 使用示例
if __name__ == "__main__":
    config = {"sample_rate": 24000, "n_fft": 1024, "hop_length": 256, "n_mels": 80}

    try:
        samples = load_training_data("data/raw_data/training_data.json")
        print(f"Loaded {len(samples)} samples from training_data.json")
    except FileNotFoundError:
        print("training_data.json not found, using example samples")
        print("Please run the audio generation script first!")

    # 处理数据
    processor = EnglishTTSDataProcessor(config)

    print(f"\nProcessing {len(samples)} samples...")

    # 并行处理（使用较少的 worker 避免内存问题）
    with ProcessPoolExecutor(max_workers=2) as executor:
        processed = list(executor.map(processor.process_single_sample, samples))

    # 过滤失败的样本
    processed = [p for p in processed if p is not None]
    print(f"Successfully processed: {len(processed)}/{len(samples)} samples")

    # 保存
    if processed:
        processor.save_processed_data(processed, "data/processed")
        print("\n✓ Data preprocessing complete!")
        print(f"  Output directory: data/processed/")
        print(f"  Files generated:")
        print(f"    - metadata.json")
        print(f"    - vocab.json")
        print(f"    - *_mel.npy")
        print(f"    - *_pitch.npy")
        print(f"    - *_energy.npy")
        print(f"    - *_wav.npy")
    else:
        print("\n✗ No samples were successfully processed!")
