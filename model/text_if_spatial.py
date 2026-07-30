"""Text_IF variant with TextSpatialAffine replacing FeatureWiseAffine.

Inherits forward() from Text_IF unchanged. Only __init__ is overridden to
swap the 4 prompt_guidance modules. The forward signature
(feat, text_features) is identical between FeatureWiseAffine and
TextSpatialAffine, so inheritance just works.
"""
import torch

from model.Text_IF_model import Text_IF
from model.text_spatial_affine import TextSpatialAffine


class TextIFSpatial(Text_IF):
    """Text_IF with spatial cross-attention modulation at decoder L1-L4.

    Args:
        model_clip: CLIP model (frozen)
        dim: base channel dim (default 16, matches Text_IF default)
        num_heads: attention heads in TextSpatialAffine (default 4)
        gate_scale: bounds the spatial gate in TextSpatialAffine to
            [1 - s, 1 + s]. Default 0.3 gives the spatial path ~3x more
            multiplicative influence than the original 0.1, providing a
            stronger gradient signal for attention to learn meaningful
            spatial patterns (needed when supervising attention).
    """

    def __init__(self, model_clip, dim=16, num_heads=4, gate_scale=0.3):
        super().__init__(model_clip, dim=dim)
        # Replace the 4 prompt_guidance modules.
        # Channel dims match those in Text_IF.__init__:
        #   L1: dim * 2**0, L2: dim * 2**1, L3: dim * 2**2, L4: dim * 2**3
        self.prompt_guidance_1 = TextSpatialAffine(
            text_dim=512, feat_channels=dim * 2 ** 0, num_heads=num_heads,
            gate_scale=gate_scale)
        self.prompt_guidance_2 = TextSpatialAffine(
            text_dim=512, feat_channels=dim * 2 ** 1, num_heads=num_heads,
            gate_scale=gate_scale)
        self.prompt_guidance_3 = TextSpatialAffine(
            text_dim=512, feat_channels=dim * 2 ** 2, num_heads=num_heads,
            gate_scale=gate_scale)
        self.prompt_guidance_4 = TextSpatialAffine(
            text_dim=512, feat_channels=dim * 2 ** 3, num_heads=num_heads,
            gate_scale=gate_scale)

    def forward_with_attn(self, inp_img_A, inp_img_B, text):
        """Forward pass that also returns per-level attention maps.

        Mirrors the base Text_IF.forward chain exactly, but calls each
        prompt_guidance_X with return_attn=True and collects the attention
        maps. Usable both for visualization (callers wrap in @torch.no_grad)
        and for training (attention maps are differentiable, enabling
        attention-supervision losses).

        Args:
            inp_img_A: [B, 3, H, W] visible image (per base convention I_A=vis)
            inp_img_B: [B, 3, H, W] infrared image
            text: [B, 77] CLIP tokens

        Returns:
            out: [B, 3, H, W] fused image (identical to forward())
            attn: dict {'L1','L2','L3','L4'}, each [B, num_heads, H_l, W_l]
                  softmax-normalized attention from text query to image keys
        """
        b = inp_img_A.shape[0]
        text_features = self.get_text_feature(text.expand(b, -1)).to(inp_img_A.dtype)

        out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A = \
            self.encoder_A(inp_img_A)
        out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B = \
            self.encoder_B(inp_img_B)

        out_enc_level4_A, out_enc_level4_B = self.cross_attention(
            out_enc_level4_A, out_enc_level4_B)
        out_enc_level4 = self.feature_fusion_4(out_enc_level4_A, out_enc_level4_B)
        out_enc_level4 = self.attention_spatial(out_enc_level4)

        out_enc_level4, attn4 = self.prompt_guidance_4(
            out_enc_level4, text_features, return_attn=True)
        inp_dec_level4 = out_enc_level4
        out_dec_level4 = self.decoder_level4(inp_dec_level4)

        inp_dec_level3 = self.up4_3(out_dec_level4)
        inp_dec_level3, attn3 = self.prompt_guidance_3(
            inp_dec_level3, text_features, return_attn=True)
        out_enc_level3 = self.feature_fusion_3(out_enc_level3_A, out_enc_level3_B)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2, attn2 = self.prompt_guidance_2(
            inp_dec_level2, text_features, return_attn=True)
        out_enc_level2 = self.feature_fusion_2(out_enc_level2_A, out_enc_level2_B)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1, attn1 = self.prompt_guidance_1(
            inp_dec_level1, text_features, return_attn=True)
        out_enc_level1 = self.feature_fusion_1(out_enc_level1_A, out_enc_level1_B)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1, {'L1': attn1, 'L2': attn2, 'L3': attn3, 'L4': attn4}
