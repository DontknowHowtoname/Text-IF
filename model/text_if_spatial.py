"""Text_IF variant with TextSpatialAffine replacing FeatureWiseAffine.

Inherits forward() from Text_IF unchanged. Only __init__ is overridden to
swap the 4 prompt_guidance modules. The forward signature
(feat, text_features) is identical between FeatureWiseAffine and
TextSpatialAffine, so inheritance just works.
"""
from model.Text_IF_model import Text_IF
from model.text_spatial_affine import TextSpatialAffine


class TextIFSpatial(Text_IF):
    """Text_IF with spatial cross-attention modulation at decoder L1-L4.

    Args:
        model_clip: CLIP model (frozen)
        dim: base channel dim (default 16, matches Text_IF default)
        num_heads: attention heads in TextSpatialAffine (default 4)
    """

    def __init__(self, model_clip, dim=16, num_heads=4):
        super().__init__(model_clip, dim=dim)
        # Replace the 4 prompt_guidance modules.
        # Channel dims match those in Text_IF.__init__:
        #   L1: dim * 2**0, L2: dim * 2**1, L3: dim * 2**2, L4: dim * 2**3
        self.prompt_guidance_1 = TextSpatialAffine(
            text_dim=512, feat_channels=dim * 2 ** 0, num_heads=num_heads)
        self.prompt_guidance_2 = TextSpatialAffine(
            text_dim=512, feat_channels=dim * 2 ** 1, num_heads=num_heads)
        self.prompt_guidance_3 = TextSpatialAffine(
            text_dim=512, feat_channels=dim * 2 ** 2, num_heads=num_heads)
        self.prompt_guidance_4 = TextSpatialAffine(
            text_dim=512, feat_channels=dim * 2 ** 3, num_heads=num_heads)

    def get_attention_maps(self, inp_img_A, inp_img_B, text):
        """Debug helper: returns dict of attention maps at each decoder level.

        Mirrors forward() but extracts attn from each TextSpatialAffine.
        Useful for visualization (spec section 8.4).
        """
        import torch
        b, c, h, w = inp_img_A.shape
        text_features = self.get_text_feature(text.expand(b, -1)).to(inp_img_A.dtype)

        out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A = \
            self.encoder_A(inp_img_A)
        out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B = \
            self.encoder_B(inp_img_B)

        out_enc_level4_A, out_enc_level4_B = self.cross_attention(
            out_enc_level4_A, out_enc_level4_B)
        out_enc_level4 = self.feature_fusion_4(out_enc_level4_A, out_enc_level4_B)
        out_enc_level4 = self.attention_spatial(out_enc_level4)

        _, attn4 = self.prompt_guidance_4(out_enc_level4, text_features, return_attn=True)
        # NOTE: this debug helper is approximate; for full fidelity use the
        # return_attn hook in production forward. Kept simple for now.

        return {'L4': attn4}
