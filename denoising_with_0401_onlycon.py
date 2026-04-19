"""
This is the version of compute frequency error, PSNR, SSIM, trace/F, and loss of different other_optimizers
by Chaoyan Jan. 7nd, 2026
 
without noise
try to train LBFGS with 6k 
by using 2 gpus
Jan 07

try refind the lr of optimizers to make it more smooth

this is the noise version from denoising0114.py
0218


test noise level 100
0318

test noise level 50
0401

*Uncomment if running on colab*
Set Runtime -> Change runtime type -> Under Hardware Accelerator select GPU in Google Colab
"""
from __future__ import print_function

#!git clone https://github.com/DmitryUlyanov/deep-image-prior
#!mv deep-image-prior/* ./

# Import libs
import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from models import *

import random
import torch
import torch.optim
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from utils.denoising_utils import *

from myfun_0210 import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

# 解析命令行参数：每个进程指定使用的 GPU 和优化器
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (per process)')
parser.add_argument('--gpus', type=str, default=None,
                    help='Comma-separated GPU ids for single-process multi-GPU (e.g. "0,1"). Overrides --gpu.')
parser.add_argument('--optimizer', type=str, default=None,
                    help='optimizer name to run (e.g. adam, SGD); if None, run all sequentially')
args = parser.parse_args()

def _parse_device_ids(gpu_arg, gpus_arg):
    if gpus_arg is None or str(gpus_arg).strip() == "":
        return [int(gpu_arg)]
    ids = []
    for s in str(gpus_arg).split(','):
        s = s.strip()
        if s == "":
            continue
        ids.append(int(s))
    if len(ids) == 0:
        return [int(gpu_arg)]
    return ids

def _unwrap_state_dict(sd):
    # Make checkpoints portable across DataParallel/non-DataParallel.
    if isinstance(sd, dict) and any(k.startswith('module.') for k in sd.keys()):
        return {k[len('module.'):]: v for k, v in sd.items()}
    return sd

def _load_model_state(net_, state_dict):
    target = net_.module if isinstance(net_, nn.DataParallel) else net_
    target.load_state_dict(_unwrap_state_dict(state_dict), strict=True)

def _model_state_dict(net_):
    target = net_.module if isinstance(net_, nn.DataParallel) else net_
    return target.state_dict()

device_ids = _parse_device_ids(args.gpu, args.gpus)
primary_cuda = (torch.cuda.is_available() and len(device_ids) > 0)
device = torch.device(f'cuda:{device_ids[0]}' if primary_cuda else 'cpu')
if args.gpus is not None:
    print(f'Using DataParallel device_ids={device_ids} (primary cuda:{device_ids[0]})')

imsize =-1
PLOT = False
sigma = 50
sigma_ = sigma/255.
torch.manual_seed(2025) # Fixed random seed for reproducibility

# deJPEG
# fname = 'data/denoising/snail.jpg'

## denoising
fname = 'data/denoising/F16_GT.png'
# Add synthetic noise
img_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_np = pil_to_np(img_pil)

img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
np.save('img_np.npy', img_np)
np.save('img_noisy_np.npy', img_noisy_np)
# Setup
INPUT = 'noise' # 'meshgrid'
pad = 'zero'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 0 #1./20. # set to 1./20. for sigma=

LR = 0.02 # ADAM

show_every = 1
exp_weight=0.99

num_iter = 5000  # 训练20000次迭代
input_depth = 32
figsize = 4

net = get_net(input_depth, 'skip', pad,
                skip_n33d=128,
                skip_n33u=128,
                skip_n11=4,
                num_scales=5,
                upsample_mode='identity',
                downsample_mode='identity',
                act_fun='none',
                need_bn=False)
net = net.to(device)
if torch.cuda.is_available() and len(device_ids) > 1:
    net = nn.DataParallel(net, device_ids=device_ids, output_device=device_ids[0])

net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).to(device).detach()
# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().to(device)

img_noisy_torch = np_to_torch(img_noisy_np).to(device)

# Optimize

# Define the list of optimizers to run
OPTIMIZER_LIST = ['SGD', 'LBFGS', 'adam', 'ASGD', 'AdamW', 'RAdam'] #['SGD', 'LBFGS', 'adam']#

# Define a dictionary for learning rates for each optimizer
OPTIMIZER_LRS = {
    'SGD': 5, #
    'ASGD': 1, #1, #
    'LBFGS': 0.5, #1.8, # no great than 2.1
    'adam': 0.001, 
    'AdamW': 0.001, 
    'RAdam': 0.001, 
}

results = {}

def run_optimizer(OPTIMIZER_name, current_LR):
    print(f"\n===== Running {OPTIMIZER_name.upper()} with LR={current_LR} for {num_iter} iterations =====")

    # === 检查是否存在保存的结果文件 ===
    start_iter = 0
    # i = start_iter
    save_path = f'results_wotracjf_6k_noise_relr_0401_padzero/results_{OPTIMIZER_name}.pt'
    
    resume_data = None
    optimizer_state_dict = None
    
    if os.path.exists(save_path):
        try:
            resume_data = torch.load(save_path, map_location='cpu')
            print(f'Found checkpoint: {save_path}')
            if isinstance(resume_data, dict) and 'results' in resume_data:
                start_iter = len(resume_data['results'].get('loss', []))
                print(f'Resuming from iteration {start_iter}/{num_iter}')
                if 'optimizer_state_dict' in resume_data:
                    optimizer_state_dict = resume_data['optimizer_state_dict']
                    print('Found optimizer state in checkpoint')
            else:
                # 旧格式：直接是 results 字典
                start_iter = len(resume_data.get('loss', []))
                print(f'Found old format checkpoint, starting from iteration {start_iter}')
        except Exception as e:
            print(f'Warning: Failed to load checkpoint {save_path}: {e}')
            print('Starting from scratch...')
            resume_data = None
            start_iter = 0
    else:
        print(f'No checkpoint found, starting from scratch...')

    # === 重新初始化网络与输入（每个优化器从头开始） ===
    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='identity',
                  downsample_mode='identity',
                  act_fun='none',
                  need_bn=False)
    net = net.to(device)
    if torch.cuda.is_available() and len(device_ids) > 1:
        net = nn.DataParallel(net, device_ids=device_ids, output_device=device_ids[0])
    
    # 如果存在保存的结果，加载训练后的权重
    if resume_data is not None:
        if isinstance(resume_data, dict) and 'trained_state_dict' in resume_data:
            try:
                _load_model_state(net, {k: v.to(device) for k, v in resume_data['trained_state_dict'].items()})
                print(f'Loaded trained network weights from checkpoint')
            except Exception as e:
                print(f'Warning: Failed to load network weights: {e}')
                resume_data = None
                start_iter = 0
        elif isinstance(resume_data, dict) and 'trained_state_dict' in resume_data.get('results', {}):
            try:
                _load_model_state(net, {k: v.to(device) for k, v in resume_data['results']['trained_state_dict'].items()})
                print(f'Loaded trained network weights from checkpoint (nested format)')
            except Exception as e:
                print(f'Warning: Failed to load network weights: {e}')
                resume_data = None
                start_iter = 0
    
    # 保存初始权重（如果存在checkpoint则使用checkpoint中的初始权重）
    if resume_data is not None:
        if isinstance(resume_data, dict) and 'initial_state_dict' in resume_data:
            initial_state_dict = resume_data['initial_state_dict']
        elif isinstance(resume_data, dict) and 'initial_state_dict' in resume_data.get('results', {}):
            initial_state_dict = resume_data['results']['initial_state_dict']
        else:
            initial_state_dict = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    else:
        initial_state_dict = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    # 从checkpoint恢复历史数据（保留所有历史数据，不截断）
    if resume_data is not None:
        if isinstance(resume_data, dict) and 'results' in resume_data:
            results_data = resume_data['results']
        else:
            results_data = resume_data
        
        # 保留所有历史数据，不要截断！这样最终结果会包含完整的0-6000次迭代
        loss_values = list(results_data.get('loss', []))
        psnr_values = list(results_data.get('psnr', []))
        psnr_sm_values = list(results_data.get('psnr_sm', []))
        metric_sens = list(results_data.get('sens', []))
        metric_jf = list(results_data.get('jf', []))
        metric_sens_jf = list(results_data.get('sens_jf', []))
        finalresult = list(results_data.get('finalresult', []))
        finalresultsm = list(results_data.get('finalresultsm', []))
        
        # 恢复 net_input_saved（关键：确保输入一致）
        if 'net_input_saved' in resume_data:
            net_input_saved = resume_data['net_input_saved'].to(device)
            print('Restored net_input_saved from checkpoint')
        else:
            # 如果没有保存，使用固定随机种子重新生成（确保一致性）
            torch.manual_seed(2025)
            np.random.seed(2025)
            random.seed(2025)
            net_input_saved = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).to(device).detach()
            print('Warning: net_input_saved not found in checkpoint, regenerated with fixed seed')
        
        # 恢复 out_avg（关键：确保平滑输出连续）
        if 'out_avg' in resume_data and resume_data['out_avg'] is not None:
            out_avg = resume_data['out_avg'].to(device)
            print('Restored out_avg from checkpoint')
        elif len(psnr_sm_values) > 0 and len(finalresult) > 0:
            # 如果没有保存out_avg，尝试从最后一个结果恢复（但这不是最优的）
            try:
                out_avg = finalresult[-1].to(device) if isinstance(finalresult[-1], torch.Tensor) else None
                print('Warning: out_avg not found in checkpoint, using last finalresult (may cause small discontinuity)')
            except:
                out_avg = None
        else:
            out_avg = None
        
        # 验证恢复后的网络输出（确保连续性）
        # 注意：验证时使用net_input_saved（无噪声），而checkpoint中的PSNR是基于带噪声的输入计算的
        # 所以会有差异是正常的，关键是训练时的连续性
        if len(finalresult) > 0 and len(psnr_values) > 0:
            try:
                net.eval()
                with torch.no_grad():
                    test_out = net(net_input_saved)
                    # 计算与checkpoint中最后一个输出的差异（基于无噪声输入）
                    if isinstance(finalresult[-1], torch.Tensor):
                        last_out = finalresult[-1].to(device) if not finalresult[-1].is_cuda else finalresult[-1]
                        output_diff = torch.mean((test_out - last_out) ** 2).item()
                        test_psnr = compare_psnr(img_np, test_out.detach().cpu().numpy()[0])
                        last_psnr = psnr_values[-1]
                        psnr_diff = abs(test_psnr - last_psnr)
                        print(f'Network verification: output MSE diff={output_diff:.6f}, PSNR diff={psnr_diff:.4f}')
                        print(f'  Note: Small differences are normal due to random noise in training (reg_noise_std={reg_noise_std:.4f})')
                        if output_diff > 0.01 or psnr_diff > 1.0:
                            print(f'Warning: Large difference detected - output MSE={output_diff:.6f}, PSNR diff={psnr_diff:.4f}')
                net.train()
            except Exception as e:
                print(f'Warning: Failed to verify network output: {e}')
        
        print(f'Resumed {len(loss_values)} iterations of history data')
        if len(psnr_values) > 0:
            psnr_sm_str = f'{psnr_sm_values[-1]:.4f}' if len(psnr_sm_values) > 0 else 'N/A'
            print(f'Last PSNR from checkpoint: {psnr_values[-1]:.4f}, Last PSNR_sm: {psnr_sm_str}')
    else:
        # 从头开始：使用固定随机种子生成输入
        torch.manual_seed(2025)
        np.random.seed(2025)
        random.seed(2025)
        net_input_saved = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).to(device).detach()
        
        loss_values = []
        psnr_values = []
        psnr_sm_values = []
        metrics_l1_l2 = []
        metric_freq = []
        metric_sens = []
        metric_jf = []
        metric_sens_jf = []
        metric_ntk = []
        finalresult = []
        finalresultsm = []
        out_avg = None
    
    # 初始化 noise（用于 reg_noise_std）
    noise = net_input_saved.detach().clone()

    last_net = None
    psrn_noisy_last = 0

    i = start_iter # This counter starts from checkpoint iteration
    CHECKPOINT_FREQ = 10000  # Save checkpoint every 1000 iterations
    # 初始化 net_input（会在 closure 中被修改）
    net_input = net_input_saved.detach().clone()
    
    def closure():
        nonlocal i, out_avg, psrn_noisy_last, last_net, net_input, optimizer

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        # 关键：把每次迭代的输出放在 CPU 上保存，避免 GPU 显存被 list 持续占用导致 OOM
        finalresult.append(out.detach().cpu())

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        # finalresultsm.append(out_avg.detach())

        total_loss = mse(out, img_noisy_torch)
        total_loss.backward()
        loss_values.append(total_loss.item())

        psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])
        psnr_values.append(psrn_gt)
        psnr_sm_values.append(psrn_gt_sm)
        # sens_val, fro_val = jacobian_metrics(net, net_input_saved)
        # metric_sens.append(sens_val)
        # metric_jf.append(fro_val)
        
        # Print iteration count as 1-based for user clarity
        print(f"[{OPTIMIZER_name}] Iter {i+1:05d}/{num_iter:05d} | Loss {total_loss.item():.6f} | PSNR_gt {psrn_gt:.2f} | PSNR_gt_sm {psrn_gt_sm:.2f}") # | L1/L2 {ratio:.5f}")
        
        # Periodic checkpoint saving (every 1000 iterations)
        if (i + 1) % CHECKPOINT_FREQ == 0 or (i + 1) == num_iter:
            # 确保目录存在
            checkpoint_dir = os.path.dirname(save_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            
            current_trained_state_dict = {k: v.cpu().clone() for k, v in _model_state_dict(net).items()}
            # 获取当前优化器状态
            current_optimizer_state = None
            try:
                current_optimizer_state = optimizer.state_dict()
            except:
                pass
            
            # 保存 net_input_saved 和 out_avg（关键：确保输入和输出连续）
            checkpoint_dict = {
                "results": {
                    'loss': loss_values,
                    'psnr': psnr_values,
                    'psnr_sm': psnr_sm_values,
                    'sens': metric_sens,
                    'jf': metric_jf,
                    'sens_jf': metric_sens_jf,
                    'finalresult': finalresult,
                    # 'finalresultsm': finalresultsm,
                    # 'initial_state_dict': initial_state_dict,
                    # 'trained_state_dict': current_trained_state_dict
                },
                # "initial_state_dict": initial_state_dict,
                # "trained_state_dict": current_trained_state_dict,
                # "net_input_saved": net_input_saved.cpu().clone(),  # 保存基础输入
                # "out_avg": out_avg.cpu().clone() if out_avg is not None else None  # 保存平滑输出
            }
            if current_optimizer_state is not None:
                checkpoint_dict["optimizer_state_dict"] = current_optimizer_state
            
            torch.save(checkpoint_dict, save_path)
            print(f'[Checkpoint] Saved checkpoint at iteration {i+1} to {save_path}')
        
        i += 1
        return total_loss

    p = get_params(OPT_OVER, net, net_input)
    
    # 创建优化器（在denoising1204.py中管理，以便保存和恢复状态）
    optimizer = None
    if OPTIMIZER_name == 'LBFGS':
        remaining_iter = num_iter - start_iter
        if remaining_iter > 0:
            optimizer = torch.optim.LBFGS(p, max_iter=remaining_iter, lr=current_LR, tolerance_grad=-1, tolerance_change=-1)
    else:
        opt_class = {
            'adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
            'RMSprop': torch.optim.RMSprop,
            'AdamW': torch.optim.AdamW,
            'RAdam': torch.optim.RAdam,
            'ASGD': torch.optim.ASGD
        }[OPTIMIZER_name]
        optimizer = opt_class(p, lr=current_LR)
    
    # 如果存在checkpoint，恢复优化器状态
    optimizer_state_loaded = False
    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
            print('Successfully loaded optimizer state from checkpoint')
            optimizer_state_loaded = True
        except Exception as e:
            print(f'Warning: Failed to load optimizer state: {e}')
            print('Will continue without optimizer state (may affect convergence)')
    
    # 运行优化
    remaining_iter = num_iter - start_iter
    if remaining_iter > 0:
        print(f'Starting optimization with {OPTIMIZER_name.upper()} (from iter {start_iter+1} to {num_iter})')
        if OPTIMIZER_name == 'LBFGS':
            def closure2():
                optimizer.zero_grad()
                return closure()
            
            # LBFGS优化器恢复状态后，可能会出现prev_flat_grad为None的错误
            # 这是因为LBFGS需要历史梯度信息，如果状态不完整，需要重新初始化
            try:
                # 尝试调用step，如果失败则重新初始化优化器
                optimizer.step(closure2)
            except (TypeError, AttributeError) as e:
                if "NoneType" in str(e) or "prev_flat_grad" in str(e).lower() or "sub()" in str(e):
                    print(f'Warning: LBFGS optimizer state incomplete: {e}')
                    print('Reinitializing LBFGS optimizer (will lose history but can continue training)...')
                    # 重新创建优化器（会丢失历史信息，但可以继续训练）
                    optimizer = torch.optim.LBFGS(p, max_iter=remaining_iter, lr=current_LR, tolerance_grad=-1, tolerance_change=-1)
                    optimizer.step(closure2)
                else:
                    raise
        else:
            for j in range(remaining_iter):
                optimizer.zero_grad()
                loss = closure()
                optimizer.step()
    else:
        print(f'Already completed {start_iter} iterations, target is {num_iter}. Skipping optimization.')
    
    # 保存训练后的权重
    trained_state_dict = {k: v.cpu().clone() for k, v in _model_state_dict(net).items()}
    
    # 获取最终的优化器状态
    final_optimizer_state = None
    if optimizer is not None:
        try:
            final_optimizer_state = optimizer.state_dict()
        except:
            pass

    return {
        'loss': loss_values,
        'psnr': psnr_values,
        'psnr_sm': psnr_sm_values,
        'sens': metric_sens,
        'jf': metric_jf,
        'sens_jf': metric_sens_jf,
        'finalresult': finalresult,
        # 'finalresultsm': finalresultsm,
        # 'initial_state_dict': initial_state_dict,
        # 'trained_state_dict': trained_state_dict,
        # 'optimizer_state_dict': final_optimizer_state  # 添加优化器状态到返回值
    }


# 根据命令行决定跑哪些 optimizer：如果指定了 --optimizer，则只跑一个，方便多进程多卡并行
if args.optimizer is not None:
    selected_optimizers = [args.optimizer]
else:
    selected_optimizers = OPTIMIZER_LIST

# ========== main loop for every optimizer ========== # Iterate using selected_optimizers and OPTIMIZER_LRS
for OPTIMIZER_name in selected_optimizers:
    # fix random input
    torch.manual_seed(2025)
    np.random.seed(2025)
    random.seed(2025)
    lr_for_optimizer = OPTIMIZER_LRS.get(OPTIMIZER_name, LR) # Get LR from dict, fallback to global LR
    results[OPTIMIZER_name] = run_optimizer(OPTIMIZER_name, lr_for_optimizer)

    # 如果是多进程/单优化器模式，单独把该优化器结果保存成文件，便于后续汇总画图
    if args.optimizer is not None:
        save_path = f'results_wotracjf_6k_noise_relr_0401_padzero/results_{OPTIMIZER_name}.pt'
        # 保存结果、初始权重、训练后权重和优化器状态
        save_dict = {
            "results": results[OPTIMIZER_name],
            # "initial_state_dict": results[OPTIMIZER_name]['initial_state_dict'],
            # "trained_state_dict": results[OPTIMIZER_name]['trained_state_dict']
        }
        # 如果存在优化器状态，也保存
        # if results[OPTIMIZER_name].get('optimizer_state_dict') is not None:
            # save_dict["optimizer_state_dict"] = results[OPTIMIZER_name]['optimizer_state_dict']
        torch.save(save_dict, save_path)
        print(f'Saved results for {OPTIMIZER_name} to {save_path}')


# 如果是多进程单优化器模式（即指定了 --optimizer），本次只会有一个 optimizer 的结果，
# 此时可以选择只画单条曲线；如果你更希望只做计算、后面统一画图，也可以把下面整块注释掉。
# 这里选择依然画当前进程拥有的所有 optimizer 曲线（通常就是一个）。

# ========== plot ========== # Plotting logic depends on the 'results' dictionary.
plt.figure(figsize=(18, 5))

# ---- (1) Loss ----
plt.subplot(1, 3, 1)
for opt_name, data in results.items():
    plt.plot(data['loss'], label=f'{opt_name} (LR={OPTIMIZER_LRS.get(opt_name, LR)})')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

# ---- (2) PSNR (GT) ----
plt.subplot(1, 3, 2)
for opt_name, data in results.items():
    plt.plot(data['psnr'], label=f'{opt_name} (LR={OPTIMIZER_LRS.get(opt_name, LR)})')
plt.xlabel('Iteration')
plt.ylabel('PSNR')
plt.title('PSNR (GT)')
plt.legend()
plt.grid(True)

# ---- (3) PSNR (Smoothed) ----
plt.subplot(1, 3, 3)
for opt_name, data in results.items():
    plt.plot(data['psnr_sm'], label=f'{opt_name} (LR={OPTIMIZER_LRS.get(opt_name, LR)})')
plt.xlabel('Iteration')
plt.ylabel('PSNR')
plt.title('PSNR (Smoothed)')
plt.legend()
plt.grid(True)

plt.suptitle('Image Denoising Results under Different Optimizers (Each from Scratch)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'results_wotracjf_6k_noise_relr_0401_padzero/denoising_results_{OPTIMIZER_name}.png')
plt.close()
