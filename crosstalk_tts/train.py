import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb  # 可选的实验跟踪


class CrosstalkDataset(Dataset):
    """相声数据集"""

    def __init__(self, data_dir, vocab_path, split="train"):
        self.data_dir = Path(data_dir)

        # 加载元数据
        with open(self.data_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # 加载词表
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # 划分训练/验证集
        n_samples = len(self.metadata)
        n_train = int(n_samples * 0.95)

        if split == "train":
            self.metadata = self.metadata[:n_train]
        else:
            self.metadata = self.metadata[n_train:]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        # 加载特征
        sample_id = item["id"]
        sub_dir = "_".join(sample_id.split("_")[:-1])
        mel = np.load(self.data_dir / sub_dir / f"{sample_id}_mel.npy")
        pitch = np.load(self.data_dir / sub_dir / f"{sample_id}_pitch.npy")
        energy = np.load(self.data_dir / sub_dir / f"{sample_id}_energy.npy")

        # 转换音素为ID
        phoneme_ids = [self.vocab.get(p, self.vocab["<UNK>"]) for p in item["phonemes"]]

        # 计算时长 (每个音素对应多少帧)
        duration = self._compute_duration(len(phoneme_ids), len(mel))

        return {
            "phoneme_ids": torch.LongTensor(phoneme_ids),
            "speaker_id": torch.LongTensor([item["speaker"]]),
            "emotion_id": torch.LongTensor([item["emotion"]]),
            "mel": torch.FloatTensor(mel),
            "pitch": torch.FloatTensor(pitch),
            "energy": torch.FloatTensor(energy),
            "duration": torch.FloatTensor(duration),
        }

    def _compute_duration(self, n_phonemes, n_frames):
        """简单的时长对齐 (实际应使用MFA等工具)"""
        # 平均分配
        base_duration = n_frames // n_phonemes
        remainder = n_frames % n_phonemes

        duration = [base_duration] * n_phonemes
        for i in range(remainder):
            duration[i] += 1

        return duration


def collate_fn(batch):
    """批处理整理函数"""
    # 找到最大长度
    max_phoneme_len = max(item["phoneme_ids"].size(0) for item in batch)
    max_mel_len = max(item["mel"].size(0) for item in batch)

    batch_size = len(batch)
    n_mels = batch[0]["mel"].size(1)

    # 初始化批次张量
    phoneme_ids = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)
    emotion_ids = torch.zeros(batch_size, dtype=torch.long)
    mels = torch.zeros(batch_size, max_mel_len, n_mels)
    pitches = torch.zeros(batch_size, max_mel_len)
    energies = torch.zeros(batch_size, max_mel_len)
    durations = torch.zeros(batch_size, max_phoneme_len)

    # 填充
    for i, item in enumerate(batch):
        phoneme_len = item["phoneme_ids"].size(0)
        mel_len = item["mel"].size(0)

        phoneme_ids[i, :phoneme_len] = item["phoneme_ids"]
        speaker_ids[i] = item["speaker_id"]
        emotion_ids[i] = item["emotion_id"]
        mels[i, :mel_len] = item["mel"]
        pitches[i, :mel_len] = item["pitch"]
        energies[i, :mel_len] = item["energy"]
        durations[i, :phoneme_len] = item["duration"]

    return {
        "phoneme_ids": phoneme_ids,
        "speaker_ids": speaker_ids,
        "emotion_ids": emotion_ids,
        "mels": mels,
        "pitches": pitches,
        "energies": energies,
        "durations": durations,
    }


class TTS_Loss(nn.Module):
    """TTS损失函数"""

    def __init__(self):
        super().__init__()
        self.mel_loss = nn.L1Loss()
        self.duration_loss = nn.MSELoss()
        self.pitch_loss = nn.MSELoss()
        self.energy_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        计算总损失
        """
        # Mel频谱损失
        mel_loss = self.mel_loss(predictions["mel"], targets["mels"])

        # 时长损失
        duration_loss = self.duration_loss(
            predictions["duration"], targets["durations"]
        )

        # 音高损失
        pitch_loss = self.pitch_loss(predictions["pitch"], targets["pitches"])

        # 能量损失
        energy_loss = self.energy_loss(predictions["energy"], targets["energies"])

        # 加权总损失
        total_loss = (
            mel_loss + 0.1 * duration_loss + 0.1 * pitch_loss + 0.1 * energy_loss
        )

        return {
            "total": total_loss,
            "mel": mel_loss,
            "duration": duration_loss,
            "pitch": pitch_loss,
            "energy": energy_loss,
        }


class Trainer:
    """训练器"""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01,
        )

        # 学习率调度器 (Warm-up + Cosine Decay)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config["learning_rate"],
            epochs=config["epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )

        # 混合精度训练
        self.scaler = GradScaler()

        # 损失函数
        self.criterion = TTS_Loss()

        # 初始化wandb (可选)
        if config.get("use_wandb", False):
            wandb.init(project="crosstalk-tts", config=config)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播 (混合精度)
            with autocast():
                predictions = self.model(
                    batch["phoneme_ids"],
                    batch["speaker_ids"],
                    batch["emotion_ids"],
                    duration=batch["durations"],
                    pitch=batch["pitches"],
                    energy=batch["energies"],
                )

                losses = self.criterion(predictions, batch)
                loss = losses["total"]

            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # 记录
            total_loss += loss.item()
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
                }
            )

            # 记录到wandb
            if self.config.get("use_wandb", False) and batch_idx % 100 == 0:
                wandb.log(
                    {
                        "train/total_loss": losses["total"].item(),
                        "train/mel_loss": losses["mel"].item(),
                        "train/duration_loss": losses["duration"].item(),
                        "train/pitch_loss": losses["pitch"].item(),
                        "train/energy_loss": losses["energy"].item(),
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                    }
                )

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            predictions = self.model(
                batch["phoneme_ids"],
                batch["speaker_ids"],
                batch["emotion_ids"],
                duration=batch["durations"],
                pitch=batch["pitches"],
                energy=batch["energies"],
            )

            losses = self.criterion(predictions, batch)
            total_loss += losses["total"].item()

        avg_loss = total_loss / len(self.val_loader)

        if self.config.get("use_wandb", False):
            wandb.log({"val/loss": avg_loss, "epoch": epoch})

        return avg_loss

    def save_checkpoint(self, epoch, val_loss, path):
        """保存检查点"""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
            },
            path,
        )

    def train(self):
        """完整训练流程"""
        best_val_loss = float("inf")

        for epoch in range(1, self.config["epochs"] + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"{'='*50}")

            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")

            # 验证
            val_loss = self.validate(epoch)
            print(f"Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    epoch,
                    val_loss,
                    Path(self.config["checkpoint_dir"]) / "best_model.pt",
                )
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

            # 定期保存
            if epoch % self.config["save_interval"] == 0:
                self.save_checkpoint(
                    epoch,
                    val_loss,
                    Path(self.config["checkpoint_dir"])
                    / f"checkpoint_epoch_{epoch}.pt",
                )


# 主训练脚本
if __name__ == "__main__":
    from crosstalk_tts_model import CrosstalkTTS

    # 配置
    config = {
        # 模型配置
        "vocab_size": 200,
        "n_speakers": 2,
        "n_emotions": 5,
        "encoder_hidden": 128,
        "encoder_layers": 4,
        "encoder_heads": 4,
        "decoder_hidden": 128,
        "decoder_layers": 4,
        "decoder_heads": 4,
        "style_dim": 128,
        "n_mels": 80,
        # 训练配置
        "batch_size": 4,
        "learning_rate": 1e-4,
        "epochs": 10,
        "save_interval": 10,
        "checkpoint_dir": "checkpoints",
        "use_wandb": False,
        # 数据路径
        "data_dir": "data/processed",
        "vocab_path": "data/processed/vocab.json",
    }

    # 创建检查点目录
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # 准备数据
    train_dataset = CrosstalkDataset(
        config["data_dir"], config["vocab_path"], split="train"
    )
    val_dataset = CrosstalkDataset(
        config["data_dir"], config["vocab_path"], split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 创建模型
    model = CrosstalkTTS(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 训练
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
