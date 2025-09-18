#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
# from component.svd_llama_new import SVD_LlamaSdpaAttention, SVD_LlamaMLP
# from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
# from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import * 
import transformers

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

@torch.no_grad()
def obtain_ratio(name, model, calib_loader, dev, ratio, mode='v2'):
    layers = model.model.layers
    model = model.to(dev)
    # model = model.float() # !!!!!!
    if mode == 'v2':
        print("Start obtaining the Theoretical Loss in SVD-LLM v2...")
        def hook(module, input, output):
            # o = output.detach().float().squeeze()
            # U, S, VT = torch.linalg.svd(o, full_matrices=False)
            # num_s_after_trunc = int(o.shape[-1] * o.shape[-2] * ratio / (o.shape[-1] + o.shape[-2]))
            # truc_s = S[:num_s_after_trunc]
            # truc_u = U[:, :num_s_after_trunc]
            # truc_v = VT[:num_s_after_trunc, :]
            # truc_sigma = torch.diag(truc_s)
            # # sqrtSigma = torch.sqrt(truc_sigma)
            # # print(truc_u.shape,truc_sigma.shape,truc_v.shape)
            # truc_o = truc_u @ truc_sigma @ truc_v
            # diff = torch.norm(truc_o - o, p='fro')
            # module.diff += diff
            # del U,S,VT,o,truc_s,truc_v,truc_u,truc_o

            o = output.detach().float()
            so = torch.linalg.svdvals(o)
            loss = torch.square(so)
            num_s_after_trunc = int(o.shape[-1] * o.shape[-2] * ratio / (o.shape[-1] + o.shape[-2]))
            module.diff += float(loss[num_s_after_trunc:].sum())

            del o,so,loss,num_s_after_trunc
            torch.cuda.empty_cache()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.diff = 0.0
                module.register_forward_hook(hook)
        for batch in tqdm(calib_loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module._forward_hooks.clear()
        torch.cuda.empty_cache()

        model = model.cpu()

        types = ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
        type_to_index = {type_str: idx for idx, type_str in enumerate(types)}
        loss = torch.zeros([len(types),len(layers)],dtype=torch.float,device=dev)
        r = torch.zeros([len(types),len(layers)],dtype=torch.float,device=dev)
        # importance = torch.zeros([len(types),len(layers)],dtype=torch.float,device=dev)

        for i in range(len(layers)):
            subset = find_layers(layers[i])
            for name in subset:
                j = type_to_index[name]
                loss[j,i] = subset[name].diff
                # importance[j,i] = subset[name].importance
        torch.save(loss, '/data/yanghq/datasets/test/loss.pt')
        # print(loss)
        # print(loss[0])
        print("Start Computing Ratio...")
        loss = 1 / torch.log(loss)
        for i in range(loss.shape[0]):
            for j in range(loss.shape[1]):
                r[i,j] = loss[i,j] / loss[i].sum()
        r = r* loss.shape[1] * ratio
        torch.save(r, '/data/yanghq/datasets/test/v2.pt')
        # sys.exit(0)

    elif mode == 'adasvd':
        print("Start obtaining the Importance in AdaSVD...")
        def hook(module, input, output):
            x = input[0].detach().float().squeeze()
            y = output[0].detach().float().squeeze()
            # importance = torch.cosine_similarity(x,y).sum()
            importance = torch.cosine_similarity(x,y)
            importance = torch.cos(importance).sum()
            # print(x.shape,y.shape,importance.shape)
            module.importance+=float(importance)

            torch.cuda.empty_cache()
        for name, module in model.named_modules():
            if isinstance(module, transformers.models.llama.modeling_llama.LlamaDecoderLayer):
                module.importance = 0.0
                module.register_forward_hook(hook)
        for batch in tqdm(calib_loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        for name, module in model.named_modules():
            if isinstance(module, transformers.models.llama.modeling_llama.LlamaDecoderLayer):
                module._forward_hooks.clear()
        torch.cuda.empty_cache()

        model = model.cpu()

        # types = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
        # type_to_index = {type_str: idx for idx, type_str in enumerate(types)}
        # r = torch.zeros([len(types),len(layers)],dtype=torch.float,device=dev)
        importance = torch.zeros([len(layers)],dtype=torch.float,device=dev)

        for i in range(len(layers)):
            importance[i] = layers[i].importance
        # print(loss)
        # print(loss[0])
        torch.save(importance, '/data/yanghq/datasets/test/importance.pt')
        # importance = 1/importance
        adasvd = importance/float(importance.mean())
        trr = ratio
        mrr = ratio-0.2
        # adasvd = mrr + adasvd * (trr-mrr)
        adasvd = mrr + (1-adasvd) * (trr-mrr)
        r = adasvd.repeat(7,1)

    elif mode=='enum':
        print("Start enumerating the Loss...")
        def hook(module, input, output):
            # inp = input[0].detach().float()
            # o = inp @ w.T
            # if module.svdvals is None:
            # w = module.weight.detach().float()
            o = output.detach().float()
            so = torch.linalg.svdvals(o)
            if module.svdvals is None:
                module.svdvals = so
            else:
                module.svdvals += so
            del o
            torch.cuda.empty_cache()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.svdvals = None
                module.register_forward_hook(hook)
        for batch in tqdm(calib_loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module._forward_hooks.clear()
        torch.cuda.empty_cache()

        model = model.cpu()

        types = ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
        type_to_index = {type_str: idx for idx, type_str in enumerate(types)}
        # loss = torch.zeros([len(types),len(layers)],dtype=torch.float,device=dev)
        # svdvals = [dict(),]*len(layers)
        svdvals = [{} for _ in range(len(layers))]
        r = torch.zeros([len(types),len(layers)],dtype=torch.float,device=dev)

        for i in range(len(layers)):
            subset = find_layers(layers[i])
            for name in subset:
                j = type_to_index[name]
                # loss[j,i] = subset[name].diff
                # importance[j,i] = subset[name].importance
                svdvals[i][name] = subset[name].svdvals
            print(svdvals[i])
        torch.save(svdvals, '/data/yanghq/datasets/test/svdvals.pt')
        sys.exit(0)
    
    elif mode=='test':
        print("Start testing...")
        def hook(module, input, output):
            # inp = input[0].detach().float().squeeze()
            # inp = inp @ inp.T
            # sig = torch.linalg.svdvals(inp)
            # eps = 1e-5
            # count = (sig < eps).sum()
            # print(f'shape={inp.shape}, count={count}, min={sig[-1]:.2e}')
            # del inp,sig,count,
            # torch.cuda.empty_cache()
            print('save begin')
            x = input[0].detach()
            path=f'/data/yanghq/datasets/test/x.pt'
            torch.save(x,path)
            w = module.gate_proj.weight.detach()
            path=f'/data/yanghq/datasets/test/w.pt'
            torch.save(w,path)
            print('save success')
            exit(0)
        from transformers.models.llama.modeling_llama import LlamaMLP
        for name, module in model.named_modules():
            # if isinstance(module, nn.Linear):
            if isinstance(module, LlamaMLP):
                # module.svdvals = None
                module.register_forward_hook(hook)
        for batch in tqdm(calib_loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module._forward_hooks.clear()
        torch.cuda.empty_cache()
        print("Finish testing...")
        exit(0)
    else:
        raise NotImplementedError(f'{mode} not supported yet')
    return r

def pre_update(model_name, model, calib_loader, dev, ratio, mode='o'):
    seq_len = model.seqlen
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros(
    #     (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    # )
    assert len(calib_loader)<3
    batch_size = calib_loader[0]['input_ids'].shape[0]
    # inps = torch.zeros(
    #     (batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    # )
    # cache = {'i': 0, 'position_ids': None, 'cache_position': None, 'position_embeddings[0]': None, 'position_embeddings[1]': None,}
    cache = {'i': 0, 'inp':None, 'position_ids': None, 'cache_position': None, 'position_embeddings[0]': None, 'position_embeddings[1]': None,}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            print(f"cache['i']", cache['i'])
            # print(kwargs)
            # print('inp', inp.shape)
            # # print(inp)
            # print('position_ids', kwargs['position_ids'].shape)
            # # print(kwargs['position_ids'])
            # print('cache_position', kwargs['cache_position'].shape)
            # # print(kwargs['cache_position'])
            # print('position_embeddings[0]', kwargs['position_embeddings'][0].shape)
            # # print(kwargs['position_embeddings'][0])
            # print('position_embeddings[1]', kwargs['position_embeddings'][1].shape)
            # # print(kwargs['position_embeddings'][1])
            # print('attention_mask', kwargs['attention_mask'].shape)
            # print(kwargs['attention_mask'])
            # exit(0)
            # inps = inp.cpu()
            cache['i'] += 1
            if cache['position_ids'] is None:
                cache['inp'] = inp.to(dev)
                cache['position_ids'] = kwargs['position_ids'].to(dev)
                cache['cache_position'] = kwargs['cache_position'].to(dev)
                cache['position_embeddings[0]'] = kwargs['position_embeddings'][0].to(dev)
                cache['position_embeddings[1]'] = kwargs['position_embeddings'][1].to(dev)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            # only layer[0].forward()
            pass
            # break
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    
    # attention_masks = cache['attention_mask']
    inp = cache['inp']
    position_ids = cache['position_ids']
    cache_position = cache['cache_position']
    position_embeddings = (cache['position_embeddings[0]'],cache['position_embeddings[1]'],)
    outs = torch.zeros_like(inp)

    assert seq_len == position_ids.shape[-1] == cache_position.shape[-1] == position_embeddings[0].shape[-2] == position_embeddings[1].shape[-2]

    def get_causal_mask(dtype,device,seq_len,cache_position,batch_size=1):
        # transformers==4.51.3 LlamaModel._prepare_4d_causal_attention_mask_with_cache_position()
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (seq_len, seq_len), fill_value=min_dtype, dtype=dtype, device=device
        )
        if seq_len != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(seq_len, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        return causal_mask
    # attention_mask = get_causal_mask(dtype=dtype,device=dev,seq_len=seq_len,cache_position=cache_position,batch_size=1)
    attention_mask = get_causal_mask(dtype=dtype,device=dev,seq_len=seq_len,cache_position=cache_position,batch_size=len(calib_loader))
    # print(f'attention_mask',attention_mask.shape)
    # print(f'position_ids',position_ids.shape)
    # print(f'cache_position',cache_position.shape)
    # print(f'position_embeddings',position_embeddings[0].shape,position_embeddings[1].shape)
    print(f"[PRE_UPDATE] by Output")
    if mode=='batch':
        for i in tqdm(range(len(layers))):
            layer = layers[i].to(dev)
            subset = find_layers(layer)
            def hook(module, input, output):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    dtype=module.weight.dtype
                    device=module.weight.device
                    # print('weight',type(module.weight))
                    # print('input[0]',input[0].shape)
                    # print('output',output.shape)
                    # exit(0)
                    # x = input[0].detach().float().squeeze().mean(dim=0)
                    w = module.weight.detach().float().squeeze()
                    y = output.detach().float().squeeze().mean(dim=0)
                    # print(x.max(dim=0))
                    # print(y.max(dim=0))
                    # print(x.shape,w.shape,y.shape)
                    # 1. SVD_p (y)
                    print('# 1. SVD_p (y)')
                    p = int(y.shape[0] * y.shape[1] * (ratio) / (y.shape[0] + y.shape[1]))
                    actual_rank_y = torch.linalg.matrix_rank(y).item()
                    print(f"Computed target rank p = {p}")
                    print(f"Actual rank of y = {actual_rank_y}")
                    print(f"Shape of y = {y.shape}")
                    if p >= actual_rank_y:
                        print(f"!!! Warning: Target rank p is not smaller than the actual rank of y. The update will be zero.\n\n")
                        return
                    
                    u_p, s_p, v_p = torch.linalg.svd(y,full_matrices=False)
                    u_p = u_p[:, :p]
                    s_p = torch.diag(s_p[:p])
                    v_p = v_p[:p, :]
                    y_p = u_p @ s_p @ v_p
                    # 2. m = x^+ (y_p - y)
                    print(u_p.shape,s_p.shape,v_p.shape)
                    print('# 2. m = x^+ (y_p - y)')
                    x_pinv = torch.linalg.pinv(x)
                    print(x.shape,x_pinv.shape,y.shape,y_p.shape)
                    m = x_pinv @ (y_p - y)
                    print('# 3. (a,b) = SVD_k (m)')
                    # 3. (a,b) = SVD_k (m)
                    k = int(m.shape[0] * m.shape[1] * (ratio) / (m.shape[0] + m.shape[1]))
                    u_k, s_k, v_k = torch.linalg.svd(m,full_matrices=False)
                    u_k = u_k[:, :k]
                    s_k = torch.diag(s_k[:k])
                    v_k = v_k[:k, :]
                    print(u_k.shape,s_k.shape,v_k.shape)
                    w_uv = u_k @ s_k @ v_k
                    print('# 4. w_all = w + ab')
                    # 4. w_all = w + ab
                    w_all = w + w_uv.T
                    w_all = w_all.to(dtype).to(device)
                    # print(module.weight)
                    module.weight = torch.nn.Parameter(w_all)
                    # print(module.weight)
                    print(w_uv.max(dim=0))
                    exit(0)
                    del x,w,y,
                    del u_p, s_p, v_p, y_p,
                    del x_pinv, m,
                    del u_k, s_k, v_k, w_uv
                    torch.cuda.empty_cache()
            handles = []
            for name in subset:
                # subset[name].scaling_diag_matrix = 0
                handles.append(subset[name].register_forward_hook(hook))
            outs = layer(
                inp,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
            for h in handles:
                h.remove()
            # layer = layer.cpu()
            
            layers[i] = layer.cpu()
            inp = outs
            torch.cuda.empty_cache()
        
    print("[PRE_UPDATE] Finish")
    

@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev, decomposition):
    layers = model.model.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    if decomposition=='v1':
        print("Start Cholesky Decomposition...")
    elif decomposition=='v2':
        print("Start Singualar Value Decomposition...")
    else:
        raise NotImplementedError(f'{decomposition} not supported yet')
    # for i in tqdm(range(len(layers))):
    with tqdm(total=7*len(layers), desc="Processing Layers and Subsets") as pbar:
        for i in range(len(layers)):
            layer_profile = {}
            subset = find_layers(layers[i])
            for name in subset:
                pbar.update(1)
                raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
                # print(name,raw_scaling_diag_matrix.shape)
                # raw_scaling_diag_matrix.shape == [input_size, input_size] == X @ X.T
                if decomposition=='v1':
                    try:
                        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                    except Exception as e:
                        print("Warning: eigen scaling_diag_matrix is not positive!")
                        eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                        raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                        eigenvalues = None
                        del eigenvalues
                elif decomposition=='v2':
                    # raise NotImplementedError('[YANGHQ] V2 Not Implemented Yet')
                    us,ss,vst = torch.linalg.svd(raw_scaling_diag_matrix, full_matrices=False)
                    ss = torch.diag(torch.sqrt(ss))
                    scaling_diag_matrix = us @ ss
                else:
                    raise NotImplementedError(f'{decomposition} not supported yet')
                layer_profile[name] = scaling_diag_matrix.cpu()
                scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
                del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
                torch.cuda.empty_cache()
            profiling_mat[i] = layer_profile
    return profiling_mat
        

@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev, decomposition):
    seq_len = model.seqlen
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    # cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    cache = {'i': 0, 'position_ids': None, 'cache_position': None, 'position_embeddings[0]': None, 'position_embeddings[1]': None,}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            print(cache['i'])
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['position_ids'] is None:
                cache['position_ids'] = kwargs['position_ids'].to(dev)
                cache['cache_position'] = kwargs['cache_position'].to(dev)
                cache['position_embeddings[0]'] = kwargs['position_embeddings'][0].to(dev)
                cache['position_embeddings[1]'] = kwargs['position_embeddings'][1].to(dev)
            # else:
            #     cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
            #     cache['cache_position'] = torch.cat((cache['cache_position'], kwargs['cache_position'].cpu()), dim=0)
            #     cache['position_embeddings[0]'] = torch.cat((cache['position_embeddings[0]'], kwargs['position_embeddings'][0].cpu()), dim=0)
            #     cache['position_embeddings[1]'] = torch.cat((cache['position_embeddings[1]'], kwargs['position_embeddings'][1].cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
            # break
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    
    print(cache)
    
    # attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    cache_position = cache['cache_position']
    position_embeddings = (cache['position_embeddings[0]'],cache['position_embeddings[1]'],)
    
    # print(f"seq_len: {seq_len}")
    # print(f"position_ids[0].shape[1]: {position_ids.shape[-1]}")
    # print(f"cache_position[0].shape[1]: {cache_position.shape[-1]}")
    # print(f"position_embeddings[0][0].shape[1]: {position_embeddings[0].shape[-2]}")
    # print(f"position_embeddings[0][1].shape[1]: {position_embeddings[1].shape[-2]}")

    assert seq_len == position_ids.shape[-1] == cache_position.shape[-1] == position_embeddings[0].shape[-2] == position_embeddings[1].shape[-2]

    def get_causal_mask(dtype,device,seq_len,cache_position,batch_size=1):
        # transformers==4.51.3 LlamaModel._prepare_4d_causal_attention_mask_with_cache_position()
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (seq_len, seq_len), fill_value=min_dtype, dtype=dtype, device=device
        )
        if seq_len != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(seq_len, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        return causal_mask
    attention_mask = get_causal_mask(dtype=dtype,device=dev,seq_len=seq_len,cache_position=cache_position,batch_size=1)
    # attention_masks = get_causal_mask(dtype=dtype,device=torch.device(dev),seq_len=seq_len,cache_position=cache_position,batch_size=len(calib_loader))
    
    profiling_mat = {}
    
    print('[LOW_RESOURCE] !!!')
    if decomposition=='v1':
        print("Start Cholesky Decomposition...")
    elif decomposition=='v2':
        print("Start Singualar Value Decomposition...")
    else:
        raise NotImplementedError(f'{decomposition} not supported yet')
    
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            # if "opt" not in model_name:
                # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
                # outs[j] = layer(
                #     inps[j].unsqueeze(0),
                #     attention_mask=attention_masks.to(dev),
                #     position_ids=position_ids.to(dev),
                #     cache_position=cache_position.to(dev),
                #     position_embeddings=(position_embeddings[0].to(dev),position_embeddings[1].to(dev),)
                # )[0]
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    print('[FINISH] profle_svdllm_low_resource')
    return profiling_mat
     

@torch.no_grad()
def whitening(model_name, model, profiling_mat, ratio, dev, r_tab=None):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        r = ratio if r_tab is None else r_tab[:,i].squeeze()
        types = ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
        type_to_index = {type_str: idx for idx, type_str in enumerate(types)}

        # if "llama" in model_name or "vicuna" in model_name:
        # svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
        svd_attn = SVD_LlamaAttention(config=model.config, layer_idx=i, ratio=r)
        # svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        svd_mlp = SVD_LlamaMLP(config=model.config, layer_idx=i, ratio=r)
        # elif "mistral" in model_name:
        #     svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
        #     svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        # elif 'opt' in model_name:
        #     svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        # else:
            # raise NotImplementedError
        #### Replace Attn, MLP ####
        for name in subset:
            if r_tab is not None:
                j = type_to_index[name]
                r_num = r_tab[j,i]
            else:
                r_num = ratio
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            # num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * r_num / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(torch.bfloat16)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(torch.bfloat16)
            
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False):
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, layer_idx=i, ratio=r)
            svd_mlp = SVD_LlamaMLP(config=model.config, layer_idx=i, ratio=r)
            # svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            # svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else: 
                scaling_diag_matrix = None
            gpts[name] = local_update(subset[name], scaling_diag_matrix = scaling_diag_matrix, ratio=ratio, name=name, direct_update=direct_update)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
        layer = layer.to(dev)
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  
        # trucation SVD
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2])
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"truncted error: {self.error}")
        x =  torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
        self.updated_uT = torch.linalg.lstsq(x,outs).solution
        updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"updated error: {self.updated_error}")
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
        # print(f"Finish {self.name}"
    
    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    parser.add_argument('--decomposition', type=str, default='v1', help='decomposition pattern')
    parser.add_argument('--ratio_allocation', type=str, default=None, help='ratio_allocatioon workflow')
    parser.add_argument('--update', type=str, default=None, help='weight update workflow')
    parser.add_argument('--pre', type=str, default=None, help='pre-fine-tuning')
    
    
    args = parser.parse_args()
    args.ratio = 1- args.ratio
    assert args.decomposition in ['v1','v2','rsvd','naive','asvd',]
    assert args.ratio_allocation in [None,'None','v2','adasvd','enum','test',]
    assert args.update in [None,'None','v1','adasvd',]
    assert args.pre in [None,'None','o','w',]
    
    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            if args.pre not in [None,'None',]:
                batch_size = 2
                input_ids = [d['input_ids'] for d in cali_white_data[:batch_size]]
                input_ids = torch.cat(input_ids,dim=0)
                attention_mask = [d['attention_mask'] for d in cali_white_data[:batch_size]]
                attention_mask = torch.cat(attention_mask,dim=0)
                batch_data = [{'input_ids':input_ids, 'attention_mask':attention_mask, }, ]
                # print(len(batch_data))
                # data = batch_data[0]
                # print(data['input_ids'].shape)
                # print(data['attention_mask'].shape)
                pre_update(args.model, model, batch_data, args.DEV, args.ratio, args.pre)
            if args.ratio_allocation in [None,'None']:
                r=None
            else:
                r = obtain_ratio(args.model, model, cali_white_data, args.DEV, args.ratio, args.ratio_allocation)
            # profiling_mat = profle_svdllm(args.model, model, cali_white_data, args.DEV, args.decomposition)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV, args.decomposition)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening(args.model, model, profiling_mat, ratio=args.ratio, dev=args.DEV, r_tab=r)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()  # need to set to float
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, args.DEV)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')  # fp32
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')
        model.eval()
        model = model.float()
        model = model.to(args.DEV)
        for name, param in model.named_parameters():
            print(name,param.shape,param.dtype,param.device)
        torch.cuda.empty_cache()
        if args.step == 4:
            # ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
            ppl_eval(model, tokenizer, datasets=[args.dataset], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)