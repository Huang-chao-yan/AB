import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.util.dtype import dtype_range
from skimage._shared.utils import _supported_float_type, check_shape_equality, warn

def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = _supported_float_type((image0.dtype, image1.dtype))
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1


def mean_squared_error(image0, image1):
    """
    Compute the mean-squared error between two images.

    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.

    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)

def oneovermse(image_true, image_test, *, data_range=None):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.

    Returns
    -------
    psnr : float
        The PSNR metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_psnr`` to
        ``skimage.metrics.peak_signal_noise_ratio``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """
    check_shape_equality(image_true, image_test)

    if data_range is None:
        if image_true.dtype != image_test.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "image_true.")
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = np.min(image_true), np.max(image_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "image_true has intensity values outside the range expected "
                "for its data type. Please manually specify the data_range.")
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    image_true, image_test = _as_floats(image_true, image_test)

    err = mean_squared_error(image_true, image_test)
    return 1 / err  #10 * np.log10((data_range ** 2) / err)


# 设置全局字体大小
plt.rcParams.update({'font.size': 20})

def optimize(optimizer_type, parameters, closure, LR, num_iter, start_iter=0):
    """
    Optimize function with resume support.
    
    Args:
        optimizer_type: optimizer name
        parameters: parameters to optimize
        closure: closure function
        LR: learning rate
        num_iter: total number of iterations
        start_iter: starting iteration (for resume from checkpoint)
    """
    remaining_iter = num_iter - start_iter
    if remaining_iter <= 0:
        print(f'Already completed {start_iter} iterations, target is {num_iter}. Skipping optimization.')
        return
    
    if optimizer_type == 'LBFGS':
        print(f'Starting optimization with LBFGS (from iter {start_iter+1} to {num_iter})')
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=remaining_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    else:
        opt_class = {
            'adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
            'RMSprop': torch.optim.RMSprop,
            'AdamW': torch.optim.AdamW,
            'RAdam': torch.optim.RAdam,
            # 'Adafactor': Adafactor #torch.optim.Adafactor
            'ASGD': torch.optim.ASGD
        }[optimizer_type]
        optimizer = opt_class(parameters, lr=LR)
        print(f'Starting optimization with {optimizer_type.upper()} (from iter {start_iter+1} to {num_iter})')
        for j in range(remaining_iter):
            optimizer.zero_grad()
            loss = closure()
            optimizer.step()

## PSNR difference at same loss 
def plot_psnr_diff_at_same_loss(results, markers, color_map, baseline='adam', tol=1e-3):
    """
    Plot PSNR differences at same loss between baseline and other optimizers.

    Args:
        results: dict, each optimizer has keys: ['loss'], ['psnr']
        baseline: str, baseline optimizer name
        tol: float, tolerance for matching losses
    """

    base_loss = np.array(results[baseline]['loss'])
    base_psnr = np.array(results[baseline]['psnr'])

    plt.figure(figsize=(36, 20))

    for opt_name, data in results.items():
        if opt_name == baseline:
            continue

        opt_loss = np.array(data['loss'])
        opt_psnr = np.array(data['psnr'])

        diff_curve = []
        valid_idx = []

        for i, l_base in enumerate(base_loss):

            # 找所有满足 |opt_loss - l_base| < tol 的 index
            candidates = np.where(np.abs(opt_loss - l_base)/np.minimum(opt_loss,l_base) < tol)[0]

            if len(candidates) == 0:
                continue  # 该 base loss 没有匹配点

            # 按照最接近的 loss 选
            idx = candidates[np.argmin(np.abs(opt_loss[candidates] - l_base))]
            psnr_diff = base_psnr[i] - opt_psnr[idx]
            
            diff_curve.append(psnr_diff)
            valid_idx.append(i)

        if len(diff_curve) == 0:
            print(f"[Warning] No matched loss points for {opt_name}")
            continue

        plt.plot(base_loss[valid_idx], diff_curve, label=f"{baseline} vs {opt_name}", marker=markers[opt_name], linestyle=' ',color=color_map[opt_name])
        # print(len(valid_idx))

    plt.xlabel("Loss", fontsize=50)
    plt.ylabel("PSNR Difference", fontsize=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # plt.title(f"PSNR Differences at Same Loss (tol={tol})", fontsize=25)
    plt.legend(fontsize=50)
    plt.grid(color='black', linestyle='-', linewidth=0.8, alpha=0.8)

    plt.savefig('results/psnr_diff_at_same_loss.png')
    plt.close()

def plot_psnr_diff_at_same_loss_index(results, markers, color_map, baseline='adam', tol=1e-3):
    """
    Plot PSNR differences at same loss between baseline and other optimizers.

    Args:
        results: dict, each optimizer has keys: ['loss'], ['psnr']
        baseline: str, baseline optimizer name
        tol: float, tolerance for matching losses
    """

    base_loss = np.array(results[baseline]['loss'])
    base_psnr = np.array(results[baseline]['psnr'])

    plt.figure(figsize=(28, 21))

    for opt_name, data in results.items():
        if opt_name == baseline:
            continue

        opt_loss = np.array(data['loss'])
        opt_psnr = np.array(data['psnr'])

        diff_curve = []
        valid_idx = []

        for i, l_base in enumerate(base_loss):

            # 找所有满足 |opt_loss - l_base| < tol 的 index
            candidates = np.where(np.abs(opt_loss - l_base)/np.minimum(opt_loss,l_base) < tol)[0]

            if len(candidates) == 0:
                continue  # 该 base loss 没有匹配点

            # 按照最接近的 loss 选
            idx = candidates[np.argmin(np.abs(opt_loss[candidates] - l_base))]
            psnr_diff = base_psnr[i] - opt_psnr[idx]
            
            diff_curve.append(psnr_diff)
            valid_idx.append(i)

        if len(diff_curve) == 0:
            print(f"[Warning] No matched loss points for {opt_name}")
            continue

        plt.plot(valid_idx, diff_curve, label=f"{baseline} vs {opt_name}", marker=markers[opt_name], linestyle=' ',color=color_map[opt_name])
        # print(len(valid_idx))

    plt.xlabel("Loss", fontsize=50)
    plt.ylabel("PSNR Difference", fontsize=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # plt.title(f"PSNR Differences at Same Loss (tol={tol})", fontsize=25)
    plt.legend(fontsize=50)
    plt.grid(color='black', linestyle='-', linewidth=0.8, alpha=0.8)

    plt.savefig('results/psnr_diff_at_same_loss_valid_index.png')
    plt.close()


## PSNR and Smoothed frequency difference \Vert|FGx_1|-|FGx_2|\Vert_2 at same loss
import torchvision.transforms.functional as TF
def smooth_freq_diff(x1,x2, kernel_size=5):
    """
    Smooth the FFT magnitude (log domain) before computing metrics.
    """
    xx1 = TF.gaussian_blur(x1, kernel_size=kernel_size, sigma=1.0)
    xx2 = TF.gaussian_blur(x2, kernel_size=kernel_size, sigma=1.0)
    Xf1 = torch.fft.fft2(xx1, norm='ortho')
    Xf2 = torch.fft.fft2(xx2, norm='ortho')
    Xf_mag1 = torch.abs(Xf1)
    Xf_mag2 = torch.abs(Xf2)
    log_fx1 = Xf_mag1 #torch.log1p(Xf_mag1)
    log_fx2 = Xf_mag2 #torch.log1p(Xf_mag2)
    diff = log_fx1 - log_fx2
    num_up = torch.norm(diff, p=2).item()
    num_down = np.minimum(torch.norm(log_fx1, p=2).item(),torch.norm(log_fx2, p=2).item())
    result = num_up / num_down
    return result

## SSIM vs Loss
def plot_ssim_vs_loss_all_optimizers(results, gt, markers, color_map, OPTIMIZER_LRS):
    """
    对所有 optimizer 的输出计算 SSIM 并画出 SSIM vs Loss 曲线。
    
    Args:
        results: dict, 每个 optimizer 的结果，如 results['adam']['finalresult'] 和 results['adam']['loss']
        gt: numpy array, ground truth 图像，shape (C,H,W)
    """
    plt.figure(figsize=(36, 20))
    
    for opt_name, data in results.items():
        losses = data['loss']
        outputs = data['finalresult']

        ssim_vals = []
        for out in outputs:
            img_out = out.detach().cpu().numpy()[0]
            # print(img_out.shape, gt.shape)
            # print("DEBUG inside fun:", gt.shape, img_out.shape, "min dims:", min(gt.shape[-2:]), flush=True)

            # ssim_val = compare_ssim(gt.transpose(1,2,0), img_out.transpose(1,2,0), multichannel=True, data_range=gt.max() - gt.min(), win_size=7)
            ssim_val = compare_ssim(gt.transpose(1,2,0), img_out.transpose(1,2,0), channel_axis=-1, data_range=gt.max() - gt.min(), win_size=7)
                                    
            ssim_vals.append(ssim_val)
        
        plt.plot(losses, ssim_vals, marker=markers[opt_name], linestyle=' ', label=f'{opt_name} (LR={OPTIMIZER_LRS.get(opt_name, 0)})',color=color_map[opt_name])
    
    plt.xlabel("Loss", fontsize=50)
    plt.ylabel("SSIM", fontsize=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # plt.title("SSIM vs Loss for All Optimizers", fontsize=50)
    plt.legend(fontsize=50)
    plt.grid(color='black', linestyle='-', linewidth=0.8, alpha=0.8)

    plt.savefig('results/ssim_vs_loss_agg_with.png')
    plt.close()


## frequency difference \Vert|Fx_1|-|Fx_2|\Vert_2 at same loss
def frequency_bias_diff_2norm(x1,x2):
    Xf1 = torch.fft.fft2(x1, norm='ortho')
    Xf2 = torch.fft.fft2(x2, norm='ortho')
    Xf_mag1 = torch.abs(Xf1)
    Xf_mag2 = torch.abs(Xf2)
    log_fx1 = Xf_mag1#torch.log1p(Xf_mag1)
    log_fx2 = Xf_mag2#torch.log1p(Xf_mag2)
    diff = log_fx1 - log_fx2
    result = torch.norm(diff, p=2).item()/torch.norm(log_fx1, p=2).item()
    return result

def plot_frequency_diff_at_same_loss(results, other_optimizers, markers, color_map,
                                     tol=1e-3, fun=frequency_bias_diff_2norm):
    """
    模仿 plot_psnr_diff_at_same_loss，在相同 loss 处比较 Adam 与其它优化器的频域差异。
    横轴：Adam 的 loss（只用有匹配点的那些 index）
    纵轴：频域 difference（如 \Vert|Fx_1|-|Fx_2|\Vert_2 归一化）
    """
    adam_loss = np.array(results['adam']['loss'])
    adam_imgs = results['adam']['finalresult']

    plt.figure(figsize=(36, 20))

    for name in other_optimizers:
        optimizer_loss = np.array(results[name]['loss'])
        optimizer_imgs = results[name]['finalresult']

        diff_curve = []
        valid_idx = []

        for i, l_adam in enumerate(adam_loss):
            # 与 PSNR 版本保持一致：使用相对误差并按最接近的 loss 匹配
            denom = np.minimum(optimizer_loss, l_adam)
            # 避免 0 造成除零
            denom = np.where(denom == 0, 1e-12, denom)

            rel_diff = np.abs(optimizer_loss - l_adam) / denom
            candidates = np.where(rel_diff < tol)[0]

            if len(candidates) == 0:
                continue

            idx = candidates[np.argmin(np.abs(optimizer_loss[candidates] - l_adam))]

            x1 = adam_imgs[i].detach().cpu()
            x2 = optimizer_imgs[idx].detach().cpu()

            ferror = fun(x1, x2)
            diff_curve.append(ferror)
            valid_idx.append(i)

        if len(diff_curve) == 0:
            print(f"[Warning] No matched loss points for {name} in frequency plot")
            continue

        plt.plot(adam_loss[valid_idx], diff_curve,
                 marker=markers[name], linestyle=' ',
                 label=f"Adam vs {name}", color=color_map[name])

    plt.xlabel("Loss", fontsize=50)
    plt.ylabel("Frequency Difference", fontsize=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(fontsize=50)
    plt.grid(color='black', linestyle='-', linewidth=0.8, alpha=0.8)

    plt.savefig('results/Frequency_spectra.png')
    plt.close()

'''
def plot_frequency_diff_at_same_loss(results, other_optimizers, markers, color_map,
                                     tol=1e-3, fun=frequency_bias_diff_2norm):
    """
    模仿 plot_loss_vs_spectral_residual_bias_psnr，在相同 loss 处比较 Adam 与其它优化器的频域差异和 PSNR 差异。
    1. 绘制 Loss vs Frequency Difference (所有优化器在一张图，单Y轴)
    2. 循环绘制每个优化器的双轴图 (左: Frequency Difference, 右: PSNR Difference)
    3. 绘制 Loss vs Frequency Difference & PSNR Difference (所有优化器在一张图，双Y轴)
    
    横轴：Adam 的 loss（只用有匹配点的那些 index）
    纵轴：频域 difference（如 \Vert|Fx_1|-|Fx_2|\Vert_2 归一化）和 PSNR Difference
    """
    adam_loss = np.array(results['adam']['loss'])
    adam_imgs = results['adam']['finalresult']
    adam_psnr = np.array(results['adam']['psnr'])

    # =======================================================
    # Part 1: 单轴图 - Loss vs Frequency Difference (所有优化器)
    # =======================================================
    plt.figure(figsize=(36, 20))

    for name in other_optimizers:
        optimizer_loss = np.array(results[name]['loss'])
        optimizer_imgs = results[name]['finalresult']

        diff_curve = []
        valid_idx = []

        for i, l_adam in enumerate(adam_loss):
            # 与 PSNR 版本保持一致：使用相对误差并按最接近的 loss 匹配
            denom = np.minimum(optimizer_loss, l_adam)
            # 避免 0 造成除零
            denom = np.where(denom == 0, 1e-12, denom)

            rel_diff = np.abs(optimizer_loss - l_adam) / denom
            candidates = np.where(rel_diff < tol)[0]

            if len(candidates) == 0:
                continue

            idx = candidates[np.argmin(np.abs(optimizer_loss[candidates] - l_adam))]

            x1 = adam_imgs[i].detach().cpu()
            x2 = optimizer_imgs[idx].detach().cpu()

            ferror = fun(x1, x2)
            diff_curve.append(ferror)
            valid_idx.append(i)

        if len(diff_curve) == 0:
            print(f"[Warning] No matched loss points for {name} in frequency plot")
            continue

        plt.plot(adam_loss[valid_idx], diff_curve,
                 marker=markers[name], linestyle=' ',
                 label=f"Adam vs {name}", color=color_map[name])

    plt.xlabel("Loss", fontsize=50)
    plt.ylabel("Frequency Difference", fontsize=50)
    plt.legend(fontsize=50)
    plt.grid(color='black', linestyle='-', linewidth=0.8, alpha=0.8)

    plt.tight_layout()
    plt.savefig('results/Frequency_spectra.png')
    plt.close()

    # =======================================================
    # Part 2: 循环绘制单张双轴图 (Frequency Difference & PSNR Difference per Optimizer)
    # =======================================================
    for name in other_optimizers:
        optimizer_loss = np.array(results[name]['loss'])
        optimizer_imgs = results[name]['finalresult']
        
        if 'psnr' not in results[name]:
            continue
        optimizer_psnr = np.array(results[name]['psnr'])

        freq_diff_curve = []
        psnr_diff_curve = []
        valid_idx = []

        for i, l_adam in enumerate(adam_loss):
            denom = np.minimum(optimizer_loss, l_adam)
            denom = np.where(denom == 0, 1e-12, denom)

            rel_diff = np.abs(optimizer_loss - l_adam) / denom
            candidates = np.where(rel_diff < tol)[0]

            if len(candidates) == 0:
                continue

            idx = candidates[np.argmin(np.abs(optimizer_loss[candidates] - l_adam))]

            x1 = adam_imgs[i].detach().cpu()
            x2 = optimizer_imgs[idx].detach().cpu()

            ferror = fun(x1, x2)
            psnr_diff = adam_psnr[i] - optimizer_psnr[idx]

            freq_diff_curve.append(ferror)
            psnr_diff_curve.append(psnr_diff)
            valid_idx.append(i)

        if len(freq_diff_curve) == 0:
            print(f"[Warning] No matched loss points for {name} in frequency+PSNR plot")
            continue

        fig, ax1 = plt.subplots(figsize=(36, 20))
        ax2 = ax1.twinx()

        # 左轴：Frequency Difference
        ax1.set_xlabel("Loss", fontsize=50)
        ax1.set_ylabel("Frequency Difference", color='black', fontsize=50)
        ax1.plot(adam_loss[valid_idx], freq_diff_curve, 
                 marker=markers[name], linestyle='--', 
                 color=color_map[name], label=f"Adam vs {name} (Frequency)")

        # 右轴：PSNR Difference
        ax2.set_ylabel("PSNR Difference", color='black', fontsize=50)
        ax2.plot(adam_loss[valid_idx], psnr_diff_curve, 
                 marker=markers[name], linestyle=' ', 
                 color=color_map[name], label=f"Adam vs {name} (PSNR)")

        # Legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=50)

        ax1.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/Frequency_PSNR_diff_{name}.png')
        plt.close()

    # =======================================================
    # Part 3: 所有优化器的集合双轴图 (Combined Frequency Difference & PSNR Difference)
    # =======================================================
    fig, ax1 = plt.subplots(figsize=(36, 20))
    ax2 = ax1.twinx()

    ax1.set_xlabel("Loss", fontsize=50)
    ax1.set_ylabel("Frequency Difference", fontsize=50)
    ax2.set_ylabel("PSNR Difference", fontsize=50)

    for name in other_optimizers:
        optimizer_loss = np.array(results[name]['loss'])
        optimizer_imgs = results[name]['finalresult']
        
        if 'psnr' not in results[name]:
            continue
        optimizer_psnr = np.array(results[name]['psnr'])

        freq_diff_curve = []
        psnr_diff_curve = []
        valid_idx = []

        for i, l_adam in enumerate(adam_loss):
            denom = np.minimum(optimizer_loss, l_adam)
            denom = np.where(denom == 0, 1e-12, denom)

            rel_diff = np.abs(optimizer_loss - l_adam) / denom
            candidates = np.where(rel_diff < tol)[0]

            if len(candidates) == 0:
                continue

            idx = candidates[np.argmin(np.abs(optimizer_loss[candidates] - l_adam))]

            x1 = adam_imgs[i].detach().cpu()
            x2 = optimizer_imgs[idx].detach().cpu()

            ferror = fun(x1, x2)
            psnr_diff = adam_psnr[i] - optimizer_psnr[idx]

            freq_diff_curve.append(ferror)
            psnr_diff_curve.append(psnr_diff)
            valid_idx.append(i)

        if len(freq_diff_curve) == 0:
            continue

        c = color_map[name]
        m = markers[name]

        # 左轴绘制 Frequency Difference (实线)
        ax1.plot(adam_loss[valid_idx], freq_diff_curve, 
                 marker=m, linestyle='-', linewidth=2,
                 color=c, label=f"Adam vs {name}")

        # 右轴绘制 PSNR Difference (虚线)
        ax2.plot(adam_loss[valid_idx], psnr_diff_curve, 
                 marker=m, linestyle='--', linewidth=2, alpha=0.7,
                 color=c, label=f"Adam vs {name} (PSNR)")

    # 统一图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    ax1.legend(h1, l1, loc='upper right', fontsize=14, ncol=2, framealpha=0.8)

    ax1.grid(True)
    plt.tight_layout()
    plt.savefig('results/combined_Frequency_PSNR_diff.png')
    plt.close()
'''


def residual_spectral_moment2(x, y, alpha=1.0, eps=1e-12):
    rr = x - y  # 残差
    B, C, H, W = rr.shape
    
    # 1. FFT 变换
    # 使用 norm='ortho' 保证时域和频域能量守恒
    Rf = torch.fft.fft2(rr, norm='ortho')
    # 移频：将低频移到中心
    Rf = torch.fft.fftshift(Rf, dim=(-2, -1))

    # 2. 构建频率网格
    # fftfreq 返回频率范围是 [-0.5, 0.5)，必须同样进行 fftshift 对齐中心
    fy = torch.fft.fftfreq(H, d=1.0, device=rr.device)
    fx = torch.fft.fftfreq(W, d=1.0, device=rr.device)
    
    # 修正点：对坐标进行 shift，使 (0,0) 位于中心
    fy = torch.fft.fftshift(fy)
    fx = torch.fft.fftshift(fx)
    
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing='ij')
    freq_radius = torch.sqrt(grid_x**2 + grid_y**2)

    # 3. 计算功率谱 |F|^2
    power = torch.abs(Rf)**2

    # 4. 分子：频谱加权求和
    # power 的形状是 (B, C, H, W)，freq_radius 是 (H, W)
    # 广播机制会自动处理
    weighted_power = (freq_radius**alpha) * power
    numerator = torch.sum(weighted_power)

    # 5. 分母：残差能量 ||x-y||^2
    # 根据 Parseval 定理，这等同于频域的总能量 torch.sum(power)
    denominator = torch.sum(rr**2)

    return numerator / (denominator + eps)

def compute_spectral_residual_curve(results, gt):
    """为每个 optimizer 计算并缓存谱偏置列表。"""
    for opt, data in results.items():
        # 数据结构：results['adam']['results']['finalresult']
        res_data = data #['results']
        if 'spectral_bias' in res_data:
            continue
        spec_vals = []
        for out in res_data['finalresult']:
            img = out #.detach().cpu().numpy()[0]  # (H, W)
            spec_vals.append(residual_spectral_moment2(img, gt, alpha=1, eps=1e-12)) 
        res_data['spectral_bias'] = spec_vals



def plot_loss_vs_spectral_residual_bias_psnr(results, gt, loss_key='loss'):
    """
    1. 绘制 Loss vs Spectral Bias (所有优化器在一张图，单Y轴)。
    2. 循环绘制每个优化器的双轴图 (左: Spectral Bias, 右: PSNR)。
    3. 绘制 Loss vs Spectral Bias & PSNR (所有优化器在一张图，双Y轴)。

    Args:
        results: dict，包含 results['loss'], results['spectral_bias'], results['psnr']
        gt: Ground Truth (用于计算谱偏差)
        loss_key: str，保存 loss 的键名（默认 'loss'）
    """
    # 计算谱偏差
    compute_spectral_residual_curve(results, gt)

    # ==== 颜色和标记映射 ====
    color_map = {
        'adam': 'black',
        'RAdam': 'tab:orange',
        'AdamW': 'tab:blue',
        'ASGD': 'tab:red',
        'LBFGS': 'tab:green',
        'SGD': 'tab:purple'
    }

    markers = {
        'adam': 'o',
        'RAdam': 'x',
        'AdamW': '*',
        'ASGD': '^',
        'LBFGS': 'P',
        'SGD': 's'
    }

    spectral_label = r"$\frac{\int_\omega|\omega||\mathcal{F}(x-y)(\omega)|^2 d \omega}{\|x-y\|_2^2} $"
    #r"$\int |\omega|^2 |F(\omega-y)| d\omega /\Vert \omega-y\Vert^2$"

    # =======================================================
    # Part 1: 原有的汇总图 (Loss vs Spectral Bias - 单轴)
    # =======================================================
    plt.figure(figsize=(36, 20))
    for opt, data in results.items():
        loss = np.array(data[loss_key])
        spec = np.array(data['spectral_bias'])
        # spec = np.stack(data['spectral_bias'], axis=0)  
        # spec = spec.mean(axis=1)  
        plt.plot(loss, spec, marker=markers.get(opt, 'o'), linestyle='', 
                 color=color_map.get(opt, 'tab:gray'), label=opt)

    plt.xlabel("Loss", fontsize=50)
    plt.ylabel(spectral_label, fontsize=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # plt.title("Loss vs normalized residual spectral moment", fontsize=50)
    plt.legend(fontsize=50)
    plt.grid(color='black', linestyle='-', linewidth=0.8, alpha=0.8)

    plt.tight_layout()
    plt.savefig('results/loss_psnr_vs_normalized residual spectral moment_alpha_1.png')
    plt.close()

    # =======================================================
    # Part 2: 循环绘制单张双轴图 (Spectral Bias & PSNR per Optimizer)
    # =======================================================
    for name in results.keys():
        loss = np.array(results[name][loss_key])
        spec = np.array(results[name]['spectral_bias'])
        
        if 'psnr' in results[name]:
            psnr = np.array(results[name]['psnr'])
        else:
            continue
        
        fig, ax1 = plt.subplots(figsize=(36, 20))
        ax2 = ax1.twinx()

        # 左轴：Spectral Bias
        ax1.set_xlabel("Loss", fontsize=50)
        ax1.set_ylabel(spectral_label, color='black', fontsize=50)
        ax1.plot(loss, spec, marker=markers.get(name, 'o'), linestyle='', 
                 color=color_map.get(name, 'black'), label=f"{name} (Spectral)")

        # 右轴：PSNR
        ax2.set_ylabel("PSNR", color='black', fontsize=50)
        ax2.plot(loss, psnr, marker=markers.get(name, '.'), linestyle=' ', 
                 color=color_map.get(name, 'black'), label=f"{name} PSNR")

        # Legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=50)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        # plt.grid(color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        ax1.grid(True, which='both', axis='both',
         color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        # plt.title(f"normalized residual spectral moment & PSNR vs. Loss - {name}", fontsize=25)
        plt.tight_layout()
        plt.savefig(f'results/loss_vs_normalized residual spectral moment_PSNR_{name}_alpha_1.png')
        plt.close()

    # =======================================================
    # Part 3: 新增 - 所有优化器的集合双轴图 (Combined Spectral & PSNR)
    # =======================================================
    fig, ax1 = plt.subplots(figsize=(36, 20)) #稍微加大尺寸以容纳图例
    ax2 = ax1.twinx()

    ax1.set_xlabel("Loss", fontsize=50)
    ax1.set_ylabel(spectral_label, fontsize=50)
    ax2.set_ylabel("PSNR", fontsize=50)

    for opt, data in results.items():
        if 'psnr' not in data: continue
        
        loss = np.array(data[loss_key])
        spec = np.array(data['spectral_bias'])
        psnr = np.array(data['psnr'])
        
        c = color_map.get(opt, 'gray')
        m = markers.get(opt, 'o')

        # 左轴绘制 Spectral Bias (实线)
        ax1.plot(loss, spec, marker=m, linestyle='', linewidth=2,
                 color=c, label=f"{opt}")

        # 右轴绘制 PSNR (虚线)
        ax2.plot(loss, psnr, marker=markers.get(name, '.'), linestyle=' ', linewidth=2, alpha=0.7,
                 color=c, label=f"{opt} (PSNR)")

    # 统一图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    # 按照优化器名字排序或分组图例，以免太乱
    # ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=14, ncol=2, framealpha=0.8)
    ax1.legend(h1, l1, loc='upper right', fontsize=50, ncol=2, framealpha=0.8)
    
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # plt.grid(color='black', linestyle='-', linewidth=0.8, alpha=0.8)
    ax1.grid(True, which='both', axis='both',
         color='black', linestyle='-', linewidth=0.8, alpha=0.8)
    # plt.title("Combined: Loss vs normalized residual spectral moment (Left) & PSNR (Right)", fontsize=50)
    plt.tight_layout()
    plt.savefig('results/combined_all_spectral_and_psnr_w0_fsq_nsq_alpha_1.png')
    plt.close()

def plot_loss_vs_spectral_residual_bias_psnr_singleimage(results, gt, loss_key='loss'):
    """
    1. 绘制 Loss vs Spectral Bias (所有优化器在一张图，单Y轴)。
    2. 循环绘制每个优化器的双轴图 (左: Spectral Bias, 右: PSNR)。
    3. 绘制 Loss vs Spectral Bias & PSNR (所有优化器在一张图，双Y轴)。

    Args:
        results: dict，包含 results['loss'], results['spectral_bias'], results['psnr']
        gt: 0 
        loss_key: str，保存 loss 的键名（默认 'loss'）
    """
    # 计算谱偏差
    compute_spectral_residual_curve(results, gt)

    # ==== 颜色和标记映射 ====
    color_map = {
        'adam': 'black',
        'RAdam': 'tab:orange',
        'AdamW': 'tab:blue',
        'ASGD': 'tab:red',
        'LBFGS': 'tab:green',
        'SGD': 'tab:purple'
    }

    markers = {
        'adam': 'o',
        'RAdam': 'x',
        'AdamW': '*',
        'ASGD': '^',
        'LBFGS': 'P',
        'SGD': 's'
    }

    spectral_label = r"$\frac{\int_\omega|\omega||\mathcal{F}(x)(\omega)|^2 d \omega}{\|x\|_2^2} $"
    #r"$\int |\omega|^2 |F(\omega-y)| d\omega /\Vert \omega-y\Vert^2$"

    # =======================================================
    # Part 1: 原有的汇总图 (Loss vs Spectral Bias - 单轴)
    # =======================================================
    plt.figure(figsize=(36, 20))
    for opt, data in results.items():
        loss = np.array(data[loss_key])
        spec = np.array(data['spectral_bias'])
        # spec = np.stack(data['spectral_bias'], axis=0)  
        # spec = spec.mean(axis=1)  
        plt.plot(loss, spec, marker=markers.get(opt, 'o'), linestyle='', 
                 color=color_map.get(opt, 'tab:gray'), label=opt)

    plt.xlabel("Loss", fontsize=50)
    plt.ylabel(spectral_label, fontsize=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # plt.title("Loss vs normalized residual spectral moment", fontsize=50)
    plt.legend(fontsize=50)
    plt.grid(color='black', linestyle='-', linewidth=0.8, alpha=0.8)

    plt.tight_layout()
    plt.savefig('results/loss_psnr_vs_normalized residual spectral moment_alpha_1_singleimage.png')
    plt.close()

    # =======================================================
    # Part 2: 循环绘制单张双轴图 (Spectral Bias & PSNR per Optimizer)
    # =======================================================
    for name in results.keys():
        loss = np.array(results[name][loss_key])
        spec = np.array(results[name]['spectral_bias'])
        
        if 'psnr' in results[name]:
            psnr = np.array(results[name]['psnr'])
        else:
            continue
        
        fig, ax1 = plt.subplots(figsize=(36, 20))
        ax2 = ax1.twinx()

        # 左轴：Spectral Bias
        ax1.set_xlabel("Loss", fontsize=50)
        ax1.set_ylabel(spectral_label, color='black', fontsize=50)
        ax1.plot(loss, spec, marker=markers.get(name, 'o'), linestyle=' ', 
                 color=color_map.get(name, 'black'), label=f"{name} (Spectral)")

        # 右轴：PSNR
        ax2.set_ylabel("PSNR", color='black', fontsize=50)
        ax2.plot(loss, psnr, marker=markers.get(name, '.'), linestyle=' ', 
                 color=color_map.get(name, 'black'), label=f"{name} PSNR")

        # Legend
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=50)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)
        # plt.grid(color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        ax1.grid(True, which='both', axis='both',
         color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        # plt.title(f"normalized residual spectral moment & PSNR vs. Loss - {name}", fontsize=25)
        plt.tight_layout()
        plt.savefig(f'results/loss_vs_normalized residual spectral moment_PSNR_{name}_alpha_1_singleimage.png')
        plt.close()

    # =======================================================
    # Part 3: 新增 - 所有优化器的集合双轴图 (Combined Spectral & PSNR)
    # =======================================================
    fig, ax1 = plt.subplots(figsize=(36, 20)) #稍微加大尺寸以容纳图例
    ax2 = ax1.twinx()

    ax1.set_xlabel("Loss", fontsize=50)
    ax1.set_ylabel(spectral_label, fontsize=50)
    ax2.set_ylabel("PSNR", fontsize=50)

    for opt, data in results.items():
        if 'psnr' not in data: continue
        
        loss = np.array(data[loss_key])
        spec = np.array(data['spectral_bias'])
        psnr = np.array(data['psnr'])
        
        c = color_map.get(opt, 'gray')
        m = markers.get(opt, 'o')

        # 左轴绘制 Spectral Bias (实线)
        ax1.plot(loss, spec, marker=m, linestyle=' ', linewidth=2,
                 color=c, label=f"{opt}")

        # 右轴绘制 PSNR (虚线)
        ax2.plot(loss, psnr, marker=markers.get(name, '.'), linestyle=' ', linewidth=2, alpha=0.7,
                 color=c, label=f"{opt} (PSNR)")

    # 统一图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    # 按照优化器名字排序或分组图例，以免太乱
    # ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=14, ncol=2, framealpha=0.8)
    ax1.legend(h1, l1, loc='upper right', fontsize=50, ncol=2, framealpha=0.8)
    
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    # plt.grid(color='black', linestyle='-', linewidth=0.8, alpha=0.8)
    ax1.grid(True, which='both', axis='both',
         color='black', linestyle='-', linewidth=0.8, alpha=0.8)
    # plt.title("Combined: Loss vs normalized residual spectral moment (Left) & PSNR (Right)", fontsize=50)
    plt.tight_layout()
    plt.savefig('results/combined_all_spectral_and_psnr_w0_fsq_nsq_alpha_1_singleimage.png')
    plt.close()
