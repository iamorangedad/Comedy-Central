# Example usage
if __name__ == "__main__":
    batch_size, seq_len, context_len, vocab_size, n_mel = 2, 10, 5, 100, 80
    model = AcousticModel(vocab_size=vocab_size, n_mel=n_mel)
    phonemes = torch.randint(0, vocab_size, (batch_size, seq_len))
    context = torch.randn(batch_size, context_len, 256)
    durations = torch.ones(batch_size, seq_len) * 3  # Dummy durations
    mel, dur_pred, pitch_pred, energy_pred = model(phonemes, context, durations)
    print(f"Mel shape: {mel.shape}")  # Expected: [batch, frame_len, 80]
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
