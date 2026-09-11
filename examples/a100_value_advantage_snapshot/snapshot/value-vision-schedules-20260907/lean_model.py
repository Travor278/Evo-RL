"""Text-free SigLIP value model; same used tensors and math as old vision_only."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import SiglipVisionConfig, SiglipVisionModel

ASSETS = Path('/data/experiments/value-vision-schedules-20260907/assets')

class LeanVisionValue(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_dtype = torch.float32
        self.image_resolution = (384, 384)
        vc = SiglipVisionConfig.from_dict(json.loads((ASSETS/'vision_config.json').read_text()))
        with torch.device('meta'):
            self.vision_encoder = SiglipVisionModel(vc)
            self.image_projector = torch.nn.Sequential(torch.nn.Linear(1152,512),torch.nn.GELU(),torch.nn.Dropout(0.1))
            self.final_norm = torch.nn.LayerNorm(512)
            self.value_head = torch.nn.Sequential(torch.nn.Linear(512,512),torch.nn.GELU(),torch.nn.Dropout(0.1),torch.nn.Linear(512,201))
        self.load_state_dict(load_file(str(ASSETS/'initial-seed1000.safetensors')),strict=True,assign=True)
        emb=self.vision_encoder.vision_model.embeddings
        emb.register_buffer('position_ids',torch.arange(emb.num_positions).expand((1,-1)),persistent=False)
        self.register_buffer('image_mean',torch.full((1,1,3,1,1),0.5),persistent=False)
        self.register_buffer('image_std',torch.full((1,1,3,1,1),0.5),persistent=False)
        if cfg.use_gradient_checkpointing:
            self.vision_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        # Preserve old head initialization and CPU RNG after the old constructor.
        from safetensors.torch import load_file as read_tensors
        torch.set_rng_state(read_tensors(str(ASSETS/'initial-rng.safetensors'))['cpu_rng'])
        assert sum(p.numel() for p in self.parameters()) == 429182729

    def _preprocess_images(self, images, image_attention_mask):
        b,n=images.shape[:2]
        images=images.float()/255.0 if images.dtype==torch.uint8 else images.float()
        if bool(images.max()>1) or bool(images.min()<0): images=(images/255.0).clamp(0,1)
        flat=images.reshape(b*n,*images.shape[2:])
        if flat.shape[-2:]!=self.image_resolution:
            flat=torch.nn.functional.interpolate(flat,size=self.image_resolution,mode='bilinear',align_corners=False)
        flat=(flat-self.image_mean.view(1,3,1,1))/self.image_std.view(1,3,1,1)
        flat=flat.reshape(b,n,*flat.shape[1:])
        return flat*image_attention_mask.to(flat.dtype).view(b,n,1,1,1)

    def forward(self,images,image_attention_mask):
        x=self._preprocess_images(images,image_attention_mask)
        b,n=x.shape[:2]
        features=self.vision_encoder(pixel_values=x.reshape(b*n,*x.shape[2:]).to(self.model_dtype),return_dict=True).pooler_output
        tokens=self.image_projector(features.float()).view(b,n,-1)
        mask=image_attention_mask.unsqueeze(-1).to(tokens.dtype)
        pooled=(tokens*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.value_head(self.final_norm(pooled))
