# from grok
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERTEmbeddings(nn.Module):
    def __init__(self):
        super(BERTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(178, 128)
        self.position_embeddings = nn.Embedding(512, 128)
        self.token_type_embeddings = nn.Embedding(2, 128)
        self.LayerNorm = nn.LayerNorm(128)

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1), dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class ALBERTLayer(nn.Module):
    def __init__(self):
        super(ALBERTLayer, self).__init__()
        self.full_layer_layer_norm = nn.LayerNorm(768)
        self.attention = nn.ModuleDict(
            {
                "query": nn.Linear(768, 768, bias=True, dtype=torch.float16),
                "key": nn.Linear(768, 768, bias=True, dtype=torch.float16),
                "value": nn.Linear(768, 768, bias=True, dtype=torch.float16),
                "dense": nn.Linear(768, 768, bias=True, dtype=torch.float16),
                "LayerNorm": nn.LayerNorm(768),
            }
        )
        self.ffn = nn.Linear(768, 2048, bias=True, dtype=torch.float16)
        self.ffn_output = nn.Linear(2048, 768, bias=True, dtype=torch.float16)

    def forward(self, hidden_states, attention_mask=None):
        # Attention
        query = self.attention.query(hidden_states)
        key = self.attention.key(hidden_states)
        value = self.attention.value(hidden_states)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(768.0, dtype=torch.float16)
        )
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)
        attention_output = self.attention.dense(context)
        attention_output = self.attention.LayerNorm(attention_output + hidden_states)

        # Feed-forward
        ffn_output = self.ffn(attention_output)
        ffn_output = F.gelu(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        output = self.full_layer_layer_norm(ffn_output + attention_output)
        return output


class BERTEncoder(nn.Module):
    def __init__(self):
        super(BERTEncoder, self).__init__()
        self.embedding_hidden_mapping_in = nn.Linear(128, 768)
        self.albert_layer_groups = nn.ModuleList([nn.ModuleList([ALBERTLayer()])])

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        for layer_group in self.albert_layer_groups:
            for layer in layer_group:
                hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(178, 128)
        self.cnn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(128, 128, kernel_size=5, bias=True),
                    nn.InstanceNorm1d(128, affine=True),
                )
                for _ in range(6)
            ]
        )
        self.lstms = nn.ModuleList(
            [nn.LSTM(128, 128, batch_first=True, bidirectional=True) for _ in range(5)]
            + [
                nn.LSTM(
                    128, 128, batch_first=True, bidirectional=True, dtype=torch.float16
                )
            ]
        )
        self.text_proj = nn.Linear(256, 512)

    def forward(self, input_ids):
        x = self.embedding(input_ids).transpose(1, 2)
        for cnn_layer in self.cnn:
            x = cnn_layer(x)
            x = F.relu(x)
        x = x.transpose(1, 2)
        for lstm in self.lstms:
            x, _ = lstm(x)
        x = self.text_proj(x)
        return x


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.text_encoder = TextEncoder()
        self.duration_proj = nn.Linear(128, 50)

        self.F0 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(128, 128, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(128, affine=True),
                        nn.Linear(128, 256, bias=True),
                    ),
                    nn.Conv1d(128, 128, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(128, affine=True),
                        nn.Linear(128, 256, bias=True),
                    ),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 64, kernel_size=1, bias=True),
                    nn.Conv1d(128, 64, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(128, affine=True),
                        nn.Linear(128, 256, bias=True),
                    ),
                    nn.Conv1d(64, 64, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(64, affine=True),
                        nn.Linear(64, 128, bias=True),
                    ),
                    nn.Conv1d(64, 128, kernel_size=3, bias=True, dtype=torch.float16),
                ),
                nn.Sequential(
                    nn.Conv1d(64, 64, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(64, affine=True),
                        nn.Linear(64, 128, bias=True),
                    ),
                    nn.Conv1d(64, 64, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(64, affine=True),
                        nn.Linear(64, 128, bias=True),
                    ),
                ),
            ]
        )

        self.N = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(128, 128, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(128, affine=True),
                        nn.Linear(128, 256, bias=True),
                    ),
                    nn.Conv1d(128, 128, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(128, affine=True),
                        nn.Linear(128, 256, bias=True),
                    ),
                ),
                nn.Sequential(
                    nn.Conv1d(128, 64, kernel_size=1, bias=True),
                    nn.Conv1d(128, 64, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(128, affine=True),
                        nn.Linear(128, 256, bias=True),
                    ),
                    nn.Conv1d(64, 64, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(64, affine=True),
                        nn.Linear(64, 128, bias=True),
                    ),
                    nn.Conv1d(64, 128, kernel_size=3, bias=True, dtype=torch.float16),
                ),
                nn.Sequential(
                    nn.Conv1d(64, 64, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(64, affine=True),
                        nn.Linear(64, 128, bias=True),
                    ),
                    nn.Conv1d(64, 64, kernel_size=3, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(64, affine=True),
                        nn.Linear(64, 128, bias=True),
                    ),
                ),
            ]
        )

        self.F0_proj = nn.Conv2d(64, 1, kernel_size=1, bias=True)
        self.N_proj = nn.Conv2d(64, 1, kernel_size=1, bias=True)

    def forward(self, x):
        text_features = self.text_encoder(x)
        duration = self.duration_proj(text_features)

        f0_features = text_features.transpose(1, 2)
        for f0_layer in self.F0:
            f0_features = f0_layer(f0_features)
        f0 = self.F0_proj(f0_features.unsqueeze(-1)).squeeze(-1)

        n_features = text_features.transpose(1, 2)
        for n_layer in self.N:
            n_features = n_layer(n_features)
        n = self.N_proj(n_features.unsqueeze(-1)).squeeze(-1)

        return duration, f0, n


class GeneratorResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(GeneratorResidualBlock, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    bias=True,
                )
                for _ in range(3)
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    bias=True,
                )
                for _ in range(3)
            ]
        )
        self.adain1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.InstanceNorm1d(out_channels, affine=True),
                    nn.Linear(out_channels, out_channels * 2, bias=True),
                )
                for _ in range(3)
            ]
        )
        self.adain2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.InstanceNorm1d(out_channels, affine=True),
                    nn.Linear(out_channels, out_channels * 2, bias=True),
                )
                for _ in range(3)
            ]
        )
        self.alpha1 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, out_channels, 1)) for _ in range(3)]
        )
        self.alpha2 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, out_channels, 1)) for _ in range(3)]
        )

    def forward(self, x, condition):
        for conv1, conv2, adain1, adain2, alpha1, alpha2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2
        ):
            residual = x
            x = conv1(x)
            x = adain1[0](x)
            scale, bias = adain1[1](condition).chunk(2, dim=-1)
            x = x * scale.unsqueeze(-1) + bias.unsqueeze(-1)
            x = F.leaky_relu(x, negative_slope=0.2)
            x = x * alpha1

            x = conv2(x)
            x = adain2[0](x)
            scale, bias = adain2[1](condition).chunk(2, dim=-1)
            x = x * scale.unsqueeze(-1) + bias.unsqueeze(-1)
            x = F.leaky_relu(x, negative_slope=0.2)
            x = x * alpha2

            x = x + residual
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Conv1d(514, 256, kernel_size=1, bias=True),
            nn.Conv1d(514, 256, kernel_size=3, padding=1, bias=True),
            nn.Sequential(
                nn.InstanceNorm1d(514, affine=True), nn.Linear(514, 1028, bias=True)
            ),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.Sequential(
                nn.InstanceNorm1d(256, affine=True), nn.Linear(256, 512, bias=True)
            ),
        )

        self.decode = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(322, 256, kernel_size=1, bias=True),
                    nn.Conv1d(322, 256, kernel_size=3, padding=1, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(322, affine=True),
                        nn.Linear(322, 644, bias=True),
                    ),
                    nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(256, affine=True),
                        nn.Linear(256, 512, bias=True),
                    ),
                )
                for _ in range(3)
            ]
            + [
                nn.Sequential(
                    nn.Conv1d(322, 256, kernel_size=1, bias=True),
                    nn.Conv1d(322, 256, kernel_size=3, padding=1, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(322, affine=True),
                        nn.Linear(322, 644, bias=True),
                    ),
                    nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=True),
                    nn.Sequential(
                        nn.InstanceNorm1d(256, affine=True),
                        nn.Linear(256, 512, bias=True),
                    ),
                    nn.Conv1d(
                        256,
                        322,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                        dtype=torch.float16,
                    ),
                )
            ]
        )

        self.generator = nn.ModuleDict(
            {
                "m_source": nn.ModuleDict(
                    {
                        "l_linear": nn.Linear(512, 1, bias=True),
                        "l_sin_gen": nn.Module(),  # Simplified, as actual implementation may vary
                    }
                ),
                "noise_convs": nn.ModuleList(
                    [
                        nn.Conv1d(22, 128, kernel_size=12, bias=True),
                        nn.Conv1d(22, 64, kernel_size=1, bias=True),
                    ]
                ),
                "ups": nn.ModuleList(
                    [
                        nn.ConvTranspose1d(
                            128,
                            128,
                            kernel_size=20,
                            stride=10,
                            bias=True,
                            dtype=torch.float16,
                        ),
                        nn.ConvTranspose1d(
                            128,
                            64,
                            kernel_size=12,
                            stride=6,
                            bias=True,
                            dtype=torch.float16,
                        ),
                    ]
                ),
                "noise_res": nn.ModuleList(
                    [
                        GeneratorResidualBlock(128, 128, kernel_size=7),
                        GeneratorResidualBlock(64, 64, kernel_size=11),
                    ]
                ),
                "resblocks": nn.ModuleList(
                    [
                        GeneratorResidualBlock(128, 128),
                        GeneratorResidualBlock(128, 128),
                        GeneratorResidualBlock(64, 64),
                        GeneratorResidualBlock(64, 64),
                    ]
                ),
                "conv_post": nn.Conv1d(
                    64, 22, kernel_size=7, bias=True, dtype=torch.float16
                ),
                "stft": nn.ModuleDict(
                    {
                        "weight_forward_real": nn.Parameter(torch.randn(11, 1, 20)),
                        "weight_forward_imag": nn.Parameter(torch.randn(11, 1, 20)),
                        "weight_backward_real": nn.Parameter(torch.randn(11, 1, 20)),
                        "weight_backward_imag": nn.Parameter(torch.randn(11, 1, 20)),
                    }
                ),
            }
        )

        self.asr_res = nn.ModuleList([nn.Conv1d(512, 64, kernel_size=1, bias=True)])

        self.F0_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True)
        self.N_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, text_features, f0, n):
        encoded = self.encode[0](text_features.transpose(1, 2))
        for layer in self.encode[1:]:
            encoded = layer(encoded)

        decoded = encoded
        for layer in self.decode:
            decoded = layer(decoded)

        x = decoded.transpose(1, 2)
        for up in self.generator.ups:
            x = up(x)
            x = F.leaky_relu(x, negative_slope=0.2)

        for resblock in self.generator.resblocks:
            x = resblock(x, text_features)

        for noise_res in self.generator.noise_res:
            x = noise_res(x, text_features)

        for conv in self.generator.noise_convs:
            x = conv(x)
            x = F.leaky_relu(x, negative_slope=0.2)

        x = self.generator.conv_post(x)
        x = torch.tanh(x)

        f0 = self.F0_conv(f0.transpose(1, 2)).transpose(1, 2)
        n = self.N_conv(n.transpose(1, 2)).transpose(1, 2)

        return x, f0, n


class TTSModel(nn.Module):
    def __init__(self):
        super(TTSModel, self).__init__()
        self.bert = nn.ModuleDict(
            {
                "embeddings": BERTEmbeddings(),
                "encoder": BERTEncoder(),
                "bias": nn.Parameter(torch.zeros(128)),
            }
        )
        self.predictor = Predictor()
        self.decoder = Decoder()

    def forward(self, input_ids, attention_mask=None):
        x = self.bert.embeddings(input_ids)
        x = self.bert.encoder(x, attention_mask)
        x = x + self.bert.bias
        duration, f0, n = self.predictor(input_ids)
        output, f0, n = self.decoder(x, f0, n)
        return output, duration, f0, n
