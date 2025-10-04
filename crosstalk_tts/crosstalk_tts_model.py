import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Args:
            x: [B, T, d_model]
            mask: [B, T]
        """
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class TextEncoder(nn.Module):
    """文本编码器 - 轻量级Transformer"""

    def __init__(self, config):
        super().__init__()

        vocab_size = config["vocab_size"]
        d_model = config["encoder_hidden"]  # 256
        n_layers = config["encoder_layers"]  # 4
        n_heads = config["encoder_heads"]  # 4

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, d_model * 4)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, phonemes):
        """
        Args:
            phonemes: [B, T]
        Returns:
            hidden: [B, T, d_model]
            mask: [B, T]
        """
        # mask for <PAD>
        mask = (phonemes != 0).float()  # [B, T]

        # 嵌入 + 位置编码
        x = self.embedding(phonemes)  # [B, T, d_model]
        x = self.pos_encoding(x)

        # Transformer层
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        return x, mask


class StyleEncoder(nn.Module):
    """风格编码器 - 融合说话人和情感信息"""

    def __init__(self, config):
        super().__init__()

        n_speakers = config["n_speakers"]  # 2
        n_emotions = config["n_emotions"]  # 5
        style_dim = config["style_dim"]  # 128

        self.embed_dim = 64
        self.speaker_embedding = nn.Embedding(n_speakers, self.embed_dim)
        self.emotion_embedding = nn.Embedding(n_emotions, self.embed_dim)

        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.embed_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, speaker_id, emotion_id):
        """
        Args:
            speaker_id: [B]
            emotion_id: [B]
        Returns:
            style_vector: [B, style_dim]
        """
        speaker_emb = self.speaker_embedding(speaker_id)  # [B, embed_dim]
        emotion_emb = self.emotion_embedding(emotion_id)  # [B, embed_dim]

        # 拼接并融合
        combined = torch.cat([speaker_emb, emotion_emb], dim=-1)  # [B, 2*embed_dim]
        style = self.fusion(combined)  # [B, style_dim]

        return style


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class VariancePredictor(nn.Module):
    """方差预测器 (用于时长、音高、能量)"""

    def __init__(self, d_model):
        super().__init__()

        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                    nn.ReLU(),
                    Permute(0, 2, 1),
                    nn.LayerNorm(d_model),
                    Permute(0, 2, 1),
                    nn.Dropout(0.1),
                )
                for _ in range(2)
            ]
        )

        self.linear = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        """
        Args:
            x: [B, T, d_model]
            mask: [B, T]
        Returns:
            pred: [B, T]
        """
        x = x.transpose(1, 2)  # [B, d_model, T]

        for conv in self.conv_layers:
            x = conv(x)
            x = x * mask.unsqueeze(1)  # 应用mask

        x = x.transpose(1, 2)  # [B, T, d_model]
        pred = self.linear(x).squeeze(-1)  # [B, T]

        return pred


class LengthRegulator(nn.Module):
    """长度调节器 - 根据预测的时长扩展序列"""

    def forward(self, x, duration, mask):
        """
        Args:
            x: [B, T, d_model]
            duration: [B, T] 每个音素的帧数
            mask: [B, T]
        Returns:
            expanded: [B, T', d_model]
            expanded_mask: [B, T']
        """
        batch_size = x.size(0)
        max_len = int(duration.sum(dim=-1).max().item())

        expanded = []
        expanded_masks = []

        for b in range(batch_size):
            seq = []
            seq_mask = []
            for t in range(x.size(1)):
                if mask[b, t] == 0:
                    continue
                # 重复duration[b, t]次
                n_repeat = int(duration[b, t].item())
                seq.append(x[b, t : t + 1].repeat(n_repeat, 1))
                seq_mask.append(torch.ones(n_repeat))

            seq = torch.cat(seq, dim=0) if seq else torch.zeros(1, x.size(-1))
            seq_mask = torch.cat(seq_mask) if seq_mask else torch.zeros(1)

            # Padding到最大长度
            if seq.size(0) < max_len:
                pad = torch.zeros(max_len - seq.size(0), x.size(-1), device=x.device)
                seq = torch.cat([seq, pad], dim=0)
                mask_pad = torch.zeros(max_len - seq_mask.size(0), device=x.device)
                seq_mask = torch.cat([seq_mask, mask_pad], dim=0)
            else:
                seq = seq[:max_len]
                seq_mask = seq_mask[:max_len]

            expanded.append(seq)
            expanded_masks.append(seq_mask)

        expanded = torch.stack(expanded, dim=0)  # [B, max_len, d_model]
        expanded_masks = torch.stack(expanded_masks, dim=0)  # [B, max_len]

        return expanded, expanded_masks


class VarianceAdaptor(nn.Module):
    """方差适配器 - 预测时长、音高、能量"""

    def __init__(self, config):
        super().__init__()

        d_model = config["encoder_hidden"]

        # 时长预测器
        self.duration_predictor = VariancePredictor(d_model)

        # 音高预测器
        self.pitch_predictor = VariancePredictor(d_model)
        self.pitch_embedding = nn.Conv1d(1, d_model, kernel_size=9, padding=4)

        # 能量预测器
        self.energy_predictor = VariancePredictor(d_model)
        self.energy_embedding = nn.Conv1d(1, d_model, kernel_size=9, padding=4)

        self.length_regulator = LengthRegulator()

    def _style_adaptive_norm(self, x, style):
        """风格自适应归一化"""
        # 简化的AdaIN
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + 1e-5)

        # 使用风格向量调制
        style_expanded = style.unsqueeze(1)  # [B, 1, style_dim]
        # 这里简化处理,实际可以学习gamma和beta
        print(normalized.size, style_expanded.size)
        return normalized + style_expanded[..., : x.size(-1)]

    def forward(
        self, hidden, mask, style, gt_duration=None, gt_pitch=None, gt_energy=None
    ):
        """
        Args:
            hidden: [B, T, d_model] 文本隐藏状态
            mask: [B, T]
            style: [B, style_dim]
            gt_*: Ground truth (训练时使用)
        """
        # 风格调制 (通过AdaIN)
        hidden = self._style_adaptive_norm(hidden, style)

        # 预测时长
        pred_duration = self.duration_predictor(hidden, mask)  # [B, T]
        duration = gt_duration if gt_duration is not None else pred_duration

        # 长度调整 (根据时长扩展序列)
        expanded_hidden, mel_mask = self.length_regulator(hidden, duration, mask)
        # expanded_hidden: [B, T', d_model]

        # 预测音高
        pred_pitch = self.pitch_predictor(expanded_hidden, mel_mask)  # [B, T']
        pitch = gt_pitch if gt_pitch is not None else pred_pitch
        pitch_emb = self.pitch_embedding(pitch.unsqueeze(1)).transpose(1, 2)
        expanded_hidden = expanded_hidden + pitch_emb

        # 预测能量
        pred_energy = self.energy_predictor(expanded_hidden, mel_mask)  # [B, T']
        energy = gt_energy if gt_energy is not None else pred_energy
        energy_emb = self.energy_embedding(energy.unsqueeze(1)).transpose(1, 2)
        expanded_hidden = expanded_hidden + energy_emb

        return expanded_hidden, pred_duration, pred_pitch, pred_energy


class MelDecoder(nn.Module):
    """Mel频谱解码器"""

    def __init__(self, config):
        super().__init__()

        d_model = config["decoder_hidden"]  # 256
        n_layers = config["decoder_layers"]  # 4
        n_heads = config["decoder_heads"]  # 4
        n_mels = config["n_mels"]  # 80

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, d_model * 4)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.mel_linear = nn.Linear(d_model, n_mels)

    def forward(self, x, style):
        """
        Args:
            x: [B, T, d_model]
            style: [B, style_dim]
        Returns:
            mel: [B, T, n_mels]
        """
        # 风格调制
        style_expanded = style.unsqueeze(1).expand(-1, x.size(1), -1)
        x = x + style_expanded[..., : x.size(-1)]

        # Transformer解码
        mask = torch.ones(x.size(0), x.size(1), device=x.device)
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        mel = self.mel_linear(x)  # [B, T, n_mels]

        return mel


class CrosstalkTTS(nn.Module):
    """轻量级双人相声TTS模型"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.text_encoder = TextEncoder(config)
        self.style_encoder = StyleEncoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = MelDecoder(config)

    def forward(
        self, phonemes, speaker_id, emotion_id, duration=None, pitch=None, energy=None
    ):
        """
        Args:
            phonemes: [B, T] 音素序列
            speaker_id: [B] 说话人ID (0 or 1)
            emotion_id: [B] 情感ID (0-4)
            duration: [B, T] 音素时长 (训练时提供)
            pitch: [B, T'] 音高 (训练时提供)
            energy: [B, T'] 能量 (训练时提供)
        """
        # 1. 文本编码
        text_hidden, text_mask = self.text_encoder(phonemes)

        # 2. 风格编码
        style_vector = self.style_encoder(speaker_id, emotion_id)

        # 3. 方差适配器 (预测或使用ground truth)
        mel_hidden, pred_duration, pred_pitch, pred_energy = self.variance_adaptor(
            text_hidden, text_mask, style_vector, duration, pitch, energy
        )

        # 4. 解码为Mel频谱
        mel_output = self.decoder(mel_hidden, style_vector)

        return {
            "mel": mel_output,
            "duration": pred_duration,
            "pitch": pred_pitch,
            "energy": pred_energy,
        }


if __name__ == "__main__":
    # 配置示例
    config = {
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
    }

    # 创建模型
    model = CrosstalkTTS(config)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
