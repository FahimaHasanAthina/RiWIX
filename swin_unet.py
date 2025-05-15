<<<<<<< HEAD
from swin_transformer_encoder import SwinTransformerDown
from swin_transformer_decoder import SwinTransformerUp
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from timm.layers import to_2tuple
from einops import rearrange

class swin_Unet(nn.Module):

    def __init__(self, img_size=896, patch_size=4, in_chans=3, num_channels=2, embedd_dim=96, 
                depths=[2, 2, 2, 2, 2, 2], depths_decoder=[1, 2, 2, 2, 2, 2], 
                 num_heads=[3, 6, 12, 24, 48, 96],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, dim_scale = 4, final_upsample="expand_first", **kwargs):
        super().__init__()
        
        self.img_size = img_size
        self.embedd_dim = embedd_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_channels = num_channels
        self.depths = depths
        self.depths_decoder = depths_decoder
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.dim_scale =  dim_scale

        # split image into non overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=self.embedd_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
    
        # Encoder for downsampling
        self.encoder = SwinTransformerDown(img_size=self.img_size, 
                patch_size=self.patch_size, in_chans=self.in_chans, num_classes=self.in_chans,
                embedd_dim=self.embedd_dim, 
                depths=self.depths, 
                num_heads=self.num_heads,
                window_size=self.window_size, mlp_ratio=self.mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                use_checkpoint=False, num_patches=num_patches, patches_resolution=self.patches_resolution)


        # Decoder for upsampling
        # self.decoder = SwinTransformerUp(img_size=896, patch_size=4, patches_resolution=self.patches_resolution,
        #         embedd_dim=self.embedd_dim, 
        #         depths_decoder=self.depths_decoder, 
        #         num_heads=self.num_heads,
        #         window_size=self.window_size, mlp_ratio=self.mlp_ratio, 
        #         qkv_bias=qkv_bias, qk_scale=None,
        #         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        #         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        #         use_checkpoint=False, final_upsample=self.final_upsample)
        
        # final patch expanding layer  
        # self.expand = nn.Linear(self.embedd_dim, 16 * self.embedd_dim, bias=False)
        # self.norm = nn.LayerNorm(self.embedd_dim)
          
        # Output layer
        self.final_conv = nn.Conv2d(in_channels = self.embedd_dim // self.dim_scale**2, out_channels = self.num_channels, kernel_size=1, bias=False)
        
    def forward(self, img):
        x = self.patch_embed(img)

        x, skip_connections = self.encoder(x)
        

        for i in range(self.num_layers-1, -1, -1):
            # Change it back to B H W(image size) C
            B, L, C = x.shape
            num_patch_H, num_patch_W = self.img_size//self.patch_size, self.img_size//self.patch_size
            if i == 0:
                x = rearrange(x, 'b (num_patch_H num_patch_W) (p1 p2 c)-> b (num_patch_H p1) (num_patch_W p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale**2, num_patch_H=num_patch_H, num_patch_W=num_patch_W)
            else:
                x = torch.concat((x, skip_connections[i]), dim=-1)
                x = rearrange(x, 'b n (p d) -> b (n p) d', p=4)

        # x = self.decoder(x, skip_connections)     
        #  Change to B C H W
        x = rearrange(x, 'b h w c -> b c h w')
        out = self.final_conv(x)


        return out
    

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x



        
=======
from swin_transformer_encoder import SwinTransformerDown
from swin_transformer_decoder import SwinTransformerUp
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from timm.layers import to_2tuple

class swin_Unet(nn.Module):

    def __init__(self, img_size=896, patch_size=4, in_chans=3, num_classes=2, embedd_dim=96, 
                depths=[2, 2, 2, 2, 2, 2], depths_decoder=[1, 2, 2, 2, 2, 2], 
                 num_heads=[3, 6, 12, 24, 48, 96],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()
        
        self.img_size = img_size
        self.embedd_dim = embedd_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.depths = depths
        self.depths_decoder = depths_decoder
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.num_layers = len(depths)
        self.patch_norm = patch_norm

        # split image into non overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=self.embedd_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
    
        # Encoder for downsampling
        self.encoder = SwinTransformerDown(img_size=self.img_size, 
                patch_size=self.patch_size, in_chans=self.in_chans, num_classes=self.in_chans,
                embedd_dim=self.embedd_dim, 
                depths=self.depths, 
                num_heads=self.num_heads,
                window_size=self.window_size, mlp_ratio=self.mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                use_checkpoint=False, num_patches=num_patches, patches_resolution=self.patches_resolution)


        # Decoder for upsampling
        self.decoder = SwinTransformerUp(img_size=896, patch_size=4, patches_resolution=self.patches_resolution,
                embedd_dim=self.embedd_dim, 
                depths_decoder=self.depths_decoder, 
                num_heads=self.num_heads,
                window_size=self.window_size, mlp_ratio=self.mlp_ratio, 
                qkv_bias=qkv_bias, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                use_checkpoint=False, final_upsample=self.final_upsample)
                    
        # Output layer
        self.final_conv = nn.Conv2d(in_channels = self.embedd_dim, out_channels = num_classes, kernel_size=1, bias=False)

    def forward(self, img):
        x = self.patch_embed(img)

        x, skip_connections = self.encoder(x)

        # for i in range(self.num_layers):
        #     tmp_patch_res_H = self.patches_resolution[0] // (2 ** i)
        #     tmp_patch_res_W = self.patches_resolution[1] // (2 ** i)

        #     B, L, C = skip_connections[i].shape
        #     assert L == tmp_patch_res_H * tmp_patch_res_W, "input features has wrong size"
        #     skip_connections[i] = skip_connections[i].view(B, self.patch_size * tmp_patch_res_H, self.patch_size * tmp_patch_res_W, -1)
        #     skip_connections[i] = skip_connections[i].permute(0, 3, 1, 2)  # B,C,H,W

        y = self.decoder(x, skip_connections)     
        out = self.final_conv(y)


        return out
    

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x



        
>>>>>>> 768dfd014359f903f45a38782239f7452c9f8085
