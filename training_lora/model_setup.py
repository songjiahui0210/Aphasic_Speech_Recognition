import torch
from torch import nn
import numpy as np
from transformers import WhisperForConditionalGeneration

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LoRALayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.low_rank_u = nn.Parameter(torch.rand(out_features, rank))
        self.low_rank_v = nn.Parameter(torch.rand(rank, in_features))
        nn.init.kaiming_uniform_(self.low_rank_u, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.low_rank_v, a=np.sqrt(5))

    def forward(self, x):
        low_rank = self.low_rank_u @ self.low_rank_v
        return self.linear(x) + low_rank @ x

def modify_whisper_with_lora(whisper_model, layer_index, rank):
    in_features = whisper_model.encoder.layers[layer_index].linear.weight.shape[1]
    out_features = whisper_model.encoder.layers[layer_index].linear.weight.shape[0]
    lora_layer = LoRALayer(in_features, out_features, rank)
    whisper_model.encoder.layers[layer_index].linear = lora_layer

model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3')
modify_whisper_with_lora(model, layer_index=0, rank=32)
model.save_pretrained("../../models/whisper_lora")