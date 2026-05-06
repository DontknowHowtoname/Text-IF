"""
Text_IF_Recon: Extends Text-IF with FFBlock + FDBlock + dual-path reconstruction.
Original Text-IF fusion path is fully preserved.
"""
import torch
import torch.nn as nn

from model.Text_IF_model import Text_IF
from model.freefusion_blocks import FFBlock, FDBlock, ReconHead


class Text_IF_Recon(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Text_IF_Recon, self).__init__()

        # Original Text-IF model as submodule
        self.base = Text_IF(
            model_clip, inp_A_channels, inp_B_channels, out_channels,
            dim, num_blocks, num_refinement_blocks, heads,
            ffn_expansion_factor, bias, LayerNorm_type
        )

        # FFBlock fusion at encoder levels 1-3
        self.ffb_1 = FFBlock(in_channels=dim, out_channels=dim)           # L1: 48
        self.ffb_2 = FFBlock(in_channels=dim * 2, out_channels=dim * 2)   # L2: 96
        self.ffb_3 = FFBlock(in_channels=dim * 4, out_channels=dim * 4)   # L3: 192

        # FDBlock decoupling (2 instances)
        # fdb_ir:  fused - visible_features -> residual ≈ IR
        # fdb_vis: fused - IR_features      -> residual ≈ visible
        channels_3lev = [dim, dim * 2, dim * 4]  # [48, 96, 192]
        self.fdb_ir = FDBlock(channels_3lev)
        self.fdb_vis = FDBlock(channels_3lev)

        # Shared lightweight reconstruction head
        self.recon_head = ReconHead(
            in_channels=[dim * 4, dim * 2, dim],  # [192, 96, 48] deepest first
            out_channels=out_channels
        )

    def forward(self, inp_img_A, inp_img_B, text):
        b = inp_img_A.shape[0]
        text_features = self.base.get_text_feature(text.expand(b, -1)).to(inp_img_A.dtype)

        # ---- Encoder (run once) ----
        out_enc_L4_A, out_enc_L3_A, out_enc_L2_A, out_enc_L1_A = self.base.encoder_A(inp_img_A)
        out_enc_L4_B, out_enc_L3_B, out_enc_L2_B, out_enc_L1_B = self.base.encoder_B(inp_img_B)

        # ---- FFBlock fusion at levels 1-3 ----
        fus_L1 = self.ffb_1(out_enc_L1_A, out_enc_L1_B)   # [B, 48, H, W]
        fus_L2 = self.ffb_2(out_enc_L2_A, out_enc_L2_B)   # [B, 96, H/2, W/2]
        fus_L3 = self.ffb_3(out_enc_L3_A, out_enc_L3_B)   # [B, 192, H/4, W/4]
        fus_feas = [fus_L1, fus_L2, fus_L3]

        # 3-level features ordered [L1, L2, L3] for FDBlock
        enc_A_3lev = [out_enc_L1_A, out_enc_L2_A, out_enc_L3_A]  # visible [48, 96, 192]
        enc_B_3lev = [out_enc_L1_B, out_enc_L2_B, out_enc_L3_B]  # infrared [48, 96, 192]

        # ---- Direct reconstruction (encoder preservation constraint) ----
        # ReconHead expects [deepest, mid, shallowest] = [L3, L2, L1]
        recon_vis = self.recon_head([enc_A_3lev[2], enc_A_3lev[1], enc_A_3lev[0]])
        recon_ir = self.recon_head([enc_B_3lev[2], enc_B_3lev[1], enc_B_3lev[0]])

        # ---- FDBlock decoupling ----
        dec_ir_feas = self.fdb_ir(fus_feas, enc_A_3lev)    # fused - vis -> IR residual
        dec_vis_feas = self.fdb_vis(fus_feas, enc_B_3lev)  # fused - IR -> vis residual

        # ---- Decoupled reconstruction ----
        recon_dec_ir = self.recon_head([dec_ir_feas[2], dec_ir_feas[1], dec_ir_feas[0]])
        recon_dec_vis = self.recon_head([dec_vis_feas[2], dec_vis_feas[1], dec_vis_feas[0]])

        # ---- Original fusion path (unchanged from Text_IF.forward) ----
        out_enc_L4_A, out_enc_L4_B = self.base.cross_attention(out_enc_L4_A, out_enc_L4_B)
        out_enc_L4 = self.base.feature_fusion_4(out_enc_L4_A, out_enc_L4_B)
        out_enc_L4 = self.base.attention_spatial(out_enc_L4)
        out_enc_L4 = self.base.prompt_guidance_4(out_enc_L4, text_features)

        out_dec_L4 = self.base.decoder_level4(out_enc_L4)

        inp_dec_L3 = self.base.up4_3(out_dec_L4)
        inp_dec_L3 = self.base.prompt_guidance_3(inp_dec_L3, text_features)
        out_enc_L3 = self.base.feature_fusion_3(out_enc_L3_A, out_enc_L3_B)
        inp_dec_L3 = torch.cat([inp_dec_L3, out_enc_L3], 1)
        inp_dec_L3 = self.base.reduce_chan_level3(inp_dec_L3)
        out_dec_L3 = self.base.decoder_level3(inp_dec_L3)

        inp_dec_L2 = self.base.up3_2(out_dec_L3)
        inp_dec_L2 = self.base.prompt_guidance_2(inp_dec_L2, text_features)
        out_enc_L2 = self.base.feature_fusion_2(out_enc_L2_A, out_enc_L2_B)
        inp_dec_L2 = torch.cat([inp_dec_L2, out_enc_L2], 1)
        inp_dec_L2 = self.base.reduce_chan_level2(inp_dec_L2)
        out_dec_L2 = self.base.decoder_level2(inp_dec_L2)

        inp_dec_L1 = self.base.up2_1(out_dec_L2)
        inp_dec_L1 = self.base.prompt_guidance_1(inp_dec_L1, text_features)
        out_enc_L1 = self.base.feature_fusion_1(out_enc_L1_A, out_enc_L1_B)
        inp_dec_L1 = torch.cat([inp_dec_L1, out_enc_L1], 1)
        out_dec_L1 = self.base.decoder_level1(inp_dec_L1)

        fused = self.base.output(self.base.refinement(out_dec_L1))

        return fused, recon_ir, recon_vis, recon_dec_ir, recon_dec_vis
