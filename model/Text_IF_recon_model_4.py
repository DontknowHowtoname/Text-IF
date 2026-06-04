"""Text_IF_Recon v4: Iterative fusion with SAM feedback.

Pass 1: Global fusion without mask (identical to v2).
Pass 2: SAM generates mask from fused_1, then fusion runs again with mask guidance.

Encoder runs once and is cached for both passes.
Only the decoder + MaskGuidedAffine path runs multiple times.

When iterations=1 or sam_filter=None, behaves identically to v3 with mask=None.
"""
import torch
import torch.nn as nn

from model.Text_IF_model import Text_IF
from model.freefusion_blocks import FFBlock, FDBlock, ReconHead
from model.mask_guided_modules import MaskGuidedAffine


class Text_IF_Recon_v4(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 iterations=2):
        super(Text_IF_Recon_v4, self).__init__()

        self.iterations = iterations

        # Original Text-IF model as submodule
        self.base = Text_IF(
            model_clip, inp_A_channels, inp_B_channels, out_channels,
            dim, num_blocks, num_refinement_blocks, heads,
            ffn_expansion_factor, bias, LayerNorm_type
        )

        # Replace prompt_guidance at levels 2-4 with MaskGuidedAffine
        self.base.prompt_guidance_4 = MaskGuidedAffine(512, dim * 2 ** 3)
        self.base.prompt_guidance_3 = MaskGuidedAffine(512, dim * 2 ** 2)
        self.base.prompt_guidance_2 = MaskGuidedAffine(512, dim * 2 ** 1)

        # FFBlock fusion at encoder levels 1-3 (same as v2/v3)
        self.ffb_1 = FFBlock(in_channels=dim, out_channels=dim)
        self.ffb_2 = FFBlock(in_channels=dim * 2, out_channels=dim * 2)
        self.ffb_3 = FFBlock(in_channels=dim * 4, out_channels=dim * 4)

        # FDBlock decoupling (same as v2/v3)
        channels_3lev = [dim, dim * 2, dim * 4]
        self.fdb_ir = FDBlock(channels_3lev)
        self.fdb_vis = FDBlock(channels_3lev)

        # Shared reconstruction head (same as v2/v3)
        self.recon_head = ReconHead(
            in_channels=[dim * 4, dim * 2, dim],
            out_channels=out_channels
        )

    def _encode(self, inp_img_A, inp_img_B):
        """Run encoder once, cache results for all fusion passes."""
        out_enc_L4_A, out_enc_L3_A, out_enc_L2_A, out_enc_L1_A = self.base.encoder_A(inp_img_A)
        out_enc_L4_B, out_enc_L3_B, out_enc_L2_B, out_enc_L1_B = self.base.encoder_B(inp_img_B)

        # FFBlock fusion at levels 1-3
        fus_L1 = self.ffb_1(out_enc_L1_A, out_enc_L1_B)
        fus_L2 = self.ffb_2(out_enc_L2_A, out_enc_L2_B)
        fus_L3 = self.ffb_3(out_enc_L3_A, out_enc_L3_B)

        # Detached encoder features for reconstruction losses
        enc_A_3lev = [out_enc_L1_A.detach(), out_enc_L2_A.detach(), out_enc_L3_A.detach()]
        enc_B_3lev = [out_enc_L1_B.detach(), out_enc_L2_B.detach(), out_enc_L3_B.detach()]

        # FDBlock decoupling (only needs to run once)
        fus_feas = [fus_L1, fus_L2, fus_L3]
        dec_ir_feas = self.fdb_ir(fus_feas, enc_A_3lev)
        dec_vis_feas = self.fdb_vis(fus_feas, enc_B_3lev)

        # Reconstructions (shared across passes)
        recon_vis = self.recon_head([enc_A_3lev[2], enc_A_3lev[1], enc_A_3lev[0]])
        recon_ir = self.recon_head([enc_B_3lev[2], enc_B_3lev[1], enc_B_3lev[0]])
        recon_dec_ir = self.recon_head([dec_ir_feas[2], dec_ir_feas[1], dec_ir_feas[0]])
        recon_dec_vis = self.recon_head([dec_vis_feas[2], dec_vis_feas[1], dec_vis_feas[0]])

        # Cached for fusion passes
        cached = {
            'out_enc_L4_A': out_enc_L4_A,
            'out_enc_L4_B': out_enc_L4_B,
            'fus_L1': fus_L1,
            'fus_L2': fus_L2,
            'fus_L3': fus_L3,
        }
        return cached, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis

    def _fusion_pass(self, cached, text_features, mask=None):
        """Run decoder path once with optional mask."""
        # Clone L4 features to prevent in-place mutation across passes
        out_enc_L4_A = cached['out_enc_L4_A'].clone()
        out_enc_L4_B = cached['out_enc_L4_B'].clone()
        fus_L1 = cached['fus_L1']
        fus_L2 = cached['fus_L2']
        fus_L3 = cached['fus_L3']

        out_enc_L4_A, out_enc_L4_B = self.base.cross_attention(out_enc_L4_A, out_enc_L4_B)
        out_enc_L4 = self.base.feature_fusion_4(out_enc_L4_A, out_enc_L4_B)
        out_enc_L4 = self.base.attention_spatial(out_enc_L4)
        out_enc_L4 = self.base.prompt_guidance_4(out_enc_L4, text_features, mask)

        out_dec_L4 = self.base.decoder_level4(out_enc_L4)

        inp_dec_L3 = self.base.up4_3(out_dec_L4)
        inp_dec_L3 = self.base.prompt_guidance_3(inp_dec_L3, text_features, mask)
        inp_dec_L3 = torch.cat([inp_dec_L3, fus_L3], 1)
        inp_dec_L3 = self.base.reduce_chan_level3(inp_dec_L3)
        out_dec_L3 = self.base.decoder_level3(inp_dec_L3)

        inp_dec_L2 = self.base.up3_2(out_dec_L3)
        inp_dec_L2 = self.base.prompt_guidance_2(inp_dec_L2, text_features, mask)
        inp_dec_L2 = torch.cat([inp_dec_L2, fus_L2], 1)
        inp_dec_L2 = self.base.reduce_chan_level2(inp_dec_L2)
        out_dec_L2 = self.base.decoder_level2(inp_dec_L2)

        inp_dec_L1 = self.base.up2_1(out_dec_L2)
        inp_dec_L1 = self.base.prompt_guidance_1(inp_dec_L1, text_features)
        inp_dec_L1 = torch.cat([inp_dec_L1, fus_L1], 1)
        out_dec_L1 = self.base.decoder_level1(inp_dec_L1)

        fused = self.base.output(self.base.refinement(out_dec_L1))
        return fused

    def forward(self, inp_img_A, inp_img_B, text, sam_filter=None):
        """
        Args:
            inp_img_A: [B, 3, H, W] visible image
            inp_img_B: [B, 3, H, W] infrared image
            text: CLIP tokenized text [B, 77]
            sam_filter: IterativeSAMFilter instance (optional)
        Returns:
            (fused_final, fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis)
        """
        b = inp_img_A.shape[0]
        text_features = self.base.get_text_feature(text.expand(b, -1)).to(inp_img_A.dtype)

        # Encoder runs once
        cached, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis = self._encode(
            inp_img_A, inp_img_B)

        # Pass 1: no mask (global fusion)
        fused_1 = self._fusion_pass(cached, text_features, mask=None)

        if self.iterations <= 1 or sam_filter is None:
            return fused_1, fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis

        # Pass 2+: SAM generates mask from fused, then fuse again with mask
        fused_prev = fused_1
        for k in range(1, self.iterations):
            with torch.no_grad():
                mask = sam_filter(fused_prev.detach())
            fused_curr = self._fusion_pass(cached, text_features, mask=mask)
            fused_prev = fused_curr

        return fused_prev, fused_1, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis
