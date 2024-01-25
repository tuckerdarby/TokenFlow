import torch
import math

import comfy.utils
import comfy.model_management
import comfy.conds


def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
    src_cond = [c for c in cond if c.get('src_latent', False)]
    cond = [c for c in cond if not c.get('src_latent', False)]
    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1
    REMOVE = 2

    to_run = []

    for x in cond:
        p = comfy.samplers.get_area_and_mult(x, x_in, timestep)
            
        if p is None:
            continue
        to_run += [(p, COND)]

    if uncond is not None:
        for x in uncond:
            p = comfy.samplers.get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            to_run += [(p, UNCOND)]
    
    for x in src_cond:
        p = comfy.samplers.get_area_and_mult(x, x_in, timestep)
        if p is None:
            continue
        to_run += [(p, 2)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if comfy.samplers.can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = comfy.model_management.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp)//i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = comfy.samplers.cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        if control is not None:
            c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond))

        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = [c if c < 2 else 0 for c in cond_or_uncond]
        transformer_options["sigmas"] = timestep

        c['transformer_options'] = transformer_options

        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
        del input_x

        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                out_cond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
            elif cond_or_uncond[o] == UNCOND:
                out_uncond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
        del mult

    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond


def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
            uncond_ = None
        else:
            uncond_ = uncond

        cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)
        if "sampler_cfg_function" in model_options:
            args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                    "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
            cfg_result = x - model_options["sampler_cfg_function"](args)
        else:
            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                    "sigma": timestep, "model_options": model_options, "input": x}
            cfg_result = fn(args)

        return cfg_result


class CFGNoisePredictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
    def apply_model(self, x, timestep, cond, uncond, cond_scale, model_options={}, seed=None):
        out = sampling_function(self.inner_model, x, timestep, uncond, cond, cond_scale, model_options=model_options, seed=seed)
        return out
    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)
    

def wrap_model(model):
  return CFGNoisePredictor(model)
