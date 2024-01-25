import torch

from .utils import isinstance_str
from .tokenflow_attention import sa_forward
from .tokenflow_block import make_tokenflow_block
from .tokenflow_conv import set_batch_to_head_dim, set_head_to_batch_dim, conv_forward


def register_pivotal(diffusion_model, is_pivotal):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "pivotal_pass", is_pivotal)


def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)


def register_time(model, t):
    # hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
    # sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
    # -> output_blocks[3*1+1][0]
    # conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module = model.output_blocks[3*1+1][0]
    setattr(conv_module, 't', t)
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # up res
    # hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
    # sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
    # up attn
    # hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
    # sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
    # [1*3+0, 1*3+1, 1*3+2]
    # model.output_blocks[3][1]
    for res in up_res_dict:
        for block in up_res_dict[res]:
            # -> model.output_blocks[3*res+block][1].transformer_blocks[0].attn1
            # module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module = model.output_blocks[3*res+block][1].transformer_blocks[0].attn1
            setattr(module, 't', t)
            # -> model.output_blocks[3*res+block][1].transformer_blocks[0].attn2
            # module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            module = model.output_blocks[3*res+block][1].transformer_blocks[0].attn2
            setattr(module, 't', t)
    
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    # down res
    # hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
    # sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
    # down attn
    # hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
    for res in down_res_dict:
        for block in down_res_dict[res]:
            # -> model.input_blocks[3*res+block+1][1].transformer_blocks[0].attn1
            # module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            module = model.input_blocks[3*res+block+1][1].transformer_blocks[0].attn1
            setattr(module, 't', t)
            # -> model.input_blocks[3*res+block+1][1].transformer_blocks[0].attn2
            # module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            module = model.input_blocks[3*res+block+1][1].transformer_blocks[0].attn2
            setattr(module, 't', t)

    # hf_mid_res_prefix = f"mid_block.resnets.{j}."
    # sd_mid_res_prefix = f"middle_block.{2 * j}."
    
    # hf_mid_atn_prefix = "mid_block.attentions.0."
    # sd_mid_atn_prefix = "middle_block.1."    
            
    # -> model.middle_block[1].transformer_blocks[0].attn1
    # module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    module = model.middle_block[1].transformer_blocks[0].attn1
    setattr(module, 't', t)
    
    # -> model.middle_block[1].transformer_blocks[0].attn2
    # module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    module = model.middle_block[1].transformer_blocks[0].attn2
    setattr(module, 't', t)


def register_conv_injection(model, injection_schedule):
    # hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
    # sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
    # -> output_blocks[3*1+1][0]
    # conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module = model.output_blocks[3*1+1][0]
    conv_module.orig_forward = conv_module._forward
    conv_module._forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)


def register_extended_attention_pnp(model, injection_schedule):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.orig_forward = module.attn1.forward
            module.attn1.forward = sa_forward(module.attn1)
            module.attn1.head_to_batch_dim = set_head_to_batch_dim(module.attn1)
            module.attn1.batch_to_head_dim = set_batch_to_head_dim(module.attn1)
            setattr(module.attn1, 'injection_schedule', [])

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            # -> model.output_blocks[3*res+block][1]
            # module = model.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module = model.output_blocks[3*res+block][1].transformer_blocks[0].attn1
            module.orig_forward = module.forward
            module.forward = sa_forward(module)
            module.head_to_batch_dim = set_head_to_batch_dim(module)
            module.batch_to_head_dim = set_batch_to_head_dim(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_tokenflow_blocks(model: torch.nn.Module):
    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.orig_forward = module.forward
            module.forward = make_tokenflow_block(module)

    return model