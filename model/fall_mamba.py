import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from functools import partial

# Import từ các file module của bạn
from .feature_extractor import VideoFeatureExtractor, PatchEmbed
from .temporal_block import create_block

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,
        rescale_prenorm_residual=True,
        n_residuals_per_layer=2,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class FallMamba(nn.Module):
    def __init__(self, img_size=224, patch_size=16, depth=24, embed_dim=192, channels=3, 
                 num_classes=2, drop_rate=0., drop_path_rate=0.1, ssm_cfg=None, 
                 norm_epsilon=1e-5, initializer_cfg=None, fused_add_norm=True, 
                 rms_norm=True, residual_in_fp32=True, bimamba=True, kernel_size=1, 
                 num_frames=8, fc_drop_rate=0., device=None, dtype=None, 
                 use_checkpoint=False, checkpoint_num=0, max_len=5000):
        
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        
        # 1. Gọi Feature Extractor
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, kernel_size=kernel_size, in_chans=channels, embed_dim=embed_dim)
        self.video_feature_extractor = VideoFeatureExtractor(embed_dim)

        # Các token và embedding vị trí
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, (num_frames // kernel_size) * 200, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        # Classifier Head
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes)

        # 2. Xây dựng các layer Mamba (Gọi từ temporal_block.py)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList()
        for i in range(depth):
            block = create_block(
                embed_dim, ssm_cfg=ssm_cfg, norm_epsilon=norm_epsilon, rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm, layer_idx=i,
                bimamba=bimamba, drop_path=dpr[i], apply_temporal_attention=(i < 0), max_len=(num_frames*200)
            )
            self.layers.append(block)

        self.norm_f = nn.LayerNorm(embed_dim, eps=norm_epsilon)

        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

    def forward_features(self, video):
            # Tinh gọn hoàn toàn, chỉ giữ lại Video!
            B, C, T, H, W = video.shape
            video = video.view(B * T, C, H, W)
            
            video_features = self.video_feature_extractor(video)
            video_features = video_features.view(B, T, self.embed_dim).permute(0, 2, 1)
            video_features = video_features.unsqueeze(2).unsqueeze(2)
            video_features = video_features.expand(-1, -1, H, W, -1)
            video_features = video_features.permute(0, 1, 4, 2, 3)
            
            if video_features.shape[1] > 3:
                video_features = video_features[:, :3, :, :, :]
                
            x = self.patch_embed(video_features)
            
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((x[:, :x.size(1) // 2, :], cls_token, x[:, x.size(1) // 2:, :]), dim=1)
            x = x + self.pos_embed

            cls_tokens = x[:B, :1, :]
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            x = x + self.temporal_pos_embedding[:, :T, :]
            x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)

            x = self.pos_drop(x)
            
            # Đi qua các khối Mamba
            for layer in self.layers:
                x, _ = layer(x)

            x = self.norm_f(x)
            return x[:, 0, :] # Lấy output của CLS token để phân loại

        def forward(self, video):
            x = self.forward_features(video)
            x = self.head(self.head_drop(x))
            return x