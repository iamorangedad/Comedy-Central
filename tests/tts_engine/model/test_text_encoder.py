# 创建模型
model = TTSTextEncoder(
    vocab_size=5000, d_model=256, num_layers=4, num_heads=4, d_ff=1024, dropout=0.1
)

# 打印模型信息
print("=" * 60)
print("TTS Text Encoder - BERT-Tiny Architecture")
print("=" * 60)

model_size = model.get_model_size()
print(f"\n模型参数量: {model_size['total_params']:,}")
print(f"可训练参数: {model_size['trainable_params']:,}")
print(f"模型大小: {model_size['size_mb']:.2f} MB (FP32)")

# 模拟输入
batch_size = 2
seq_len = 50
input_ids = torch.randint(0, 5000, (batch_size, seq_len))

# 前向传播
with torch.no_grad():
    outputs = model(input_ids)

print("\n" + "=" * 60)
print("输出维度:")
print("=" * 60)
print(f"Encoder输出: {outputs['encoder_output'].shape}")
print(f"音高预测: {outputs['pitch'].shape}")
print(f"能量预测: {outputs['energy'].shape}")
print(f"时长预测: {outputs['duration'].shape}")
print(f"注意力层数: {len(outputs['attention_weights'])}")

# 显示各模块参数量
print("\n" + "=" * 60)
print("各模块参数量:")
print("=" * 60)

modules_params = {
    "Token Embedding": sum(p.numel() for p in model.token_embedding.parameters()),
    "Position Encoding": sum(p.numel() for p in model.position_encoding.parameters()),
    "Transformer Layers": sum(p.numel() for p in model.encoder_layers.parameters()),
    "Prosody Predictor": sum(p.numel() for p in model.prosody_predictor.parameters()),
    "Duration Predictor": sum(p.numel() for p in model.duration_predictor.parameters()),
}

for name, params in modules_params.items():
    print(f"{name:25s}: {params:>10,} ({params/model_size['total_params']*100:5.2f}%)")

print("\n" + "=" * 60)
print("核心功能:")
print("=" * 60)
print("✓ 字符/音素编码 - Token Embedding + 4层Transformer")
print("✓ 韵律预测 - 音高(Pitch) + 能量(Energy)")
print("✓ Duration预测 - 每个音素的持续时间")
print("=" * 60)
