# rl_noise_smoothing.py

import math
import os
import torch
import matplotlib.pyplot as plt
from helpers import RiemannLiouvilleFractionalIntegral
from mh.typlotlib import bool_slice, get_frames_bool, save_frames
import hydra
from mh.core import hydra_out

def preserve_shape_integral(f: torch.Tensor, dt):
    v = torch.cumulative_trapezoid(f, dx=dt, dim=0)
    v = torch.cat((v, v[-1:]))  # pad with last value
    return v

    
@hydra.main(config_path="all/main", config_name="refactor", version_base=None)
def main(cfg):
    # ----------------------
    # 1) Build a noisy test signal
    # ----------------------
    dt = 0.05               # time step
    N = 400                 # number of samples
    t = torch.arange(N, dtype=torch.float32) * dt

    # underlying “true” signal (a simple sine wave)
    # freq = 1.0            # Hz
    freq = cfg.freq
    f_true = torch.sin(2 * math.pi * cfg.freq * t)

    # add Gaussian noise
    if cfg.snr in [None, 'inf', 'Inf', 'Infinity', '∞']:
        noise_level = 0.0
    else:
        noise_level = torch.sqrt(torch.mean(f_true**2)) / cfg.snr
    f_noisy = f_true + noise_level * torch.randn_like(f_true)

    # ----------------------
    # 2) Choose fractional orders alpha to test
    # ----------------------
    alphas = torch.linspace(0,1,100)
    
    analytic_integral = 1.0 / (2*math.pi*freq) * (1.0 -torch.cos(2*math.pi*freq*t))

    # ----------------------
    # 3) Plot noisy signal + RL–smoothed versions
    # ----------------------
    plt.figure(figsize=(16, 9))
    plt.plot(t.numpy(), f_noisy.numpy(),
             color="lightgray", alpha=0.7, label="Noisy input")
    plt.plot(t.numpy(), f_true.numpy(),
             color="k", linewidth=1.5, label="True signal")

    classic_integral = RiemannLiouvilleFractionalIntegral(alpha=1.0, dt=dt, max_length=N)
    res = []
    for alpha in alphas:
        num_classical_applications = 0
        while alpha > 1.0:
            alpha -= 1.0
            num_classical_applications += 1
        rl = RiemannLiouvilleFractionalIntegral(alpha=alpha, dt=dt, max_length=N)
        f_smooth = rl(f_noisy)
        for _ in range(num_classical_applications):
            f_smooth = preserve_shape_integral(f_smooth, dt)
        # plt.plot(t.numpy(), f_smooth.numpy(),
        #          label=f"alpha = {alpha}")
        res.append(f_smooth)

    # pull figsize/dpi from cfg
    fig, ax = plt.subplots(
        figsize=tuple(cfg.out.figsize),
        dpi=cfg.out.get("dpi", 100),
    )

    def plotter(*, data, idx, fig, axes):
        axes.clear()
        # plot true / noisy / analytic once
        axes.plot(t, f_true,    color="violet", linewidth=1.5, label="True signal")
        axes.plot(t, f_noisy,   'k.', alpha=0.3,         label="Noisy input")
        axes.plot(t, analytic_integral,
                  color="orange", linewidth=1.5,     label="Analytic integral")

        # plot current RL smooth
        alpha = alphas[idx[0]].item()
        axes.plot(t, data[idx[0]],
                  label=f"α = {alpha:.2f}")

        axes.set_title(f"RL–smoothed signals: SNR={cfg.snr}", fontsize=20)
        axes.set_xlabel("t")
    
        # compute y‐limits just once per frame
        all_mins = torch.minimum(analytic_integral.min(),
                                 torch.tensor([d.min() for d in data]).min())
        all_maxs = torch.maximum(analytic_integral.max(),
                                 torch.tensor([d.max() for d in data]).max())
        axes.set_ylim(all_mins.item(), all_maxs.item())

        axes.legend(**cfg.out.legend)
        return None  # no state carried between calls

    # now hand fig/ax into the typlotlib machinery
    iter = bool_slice(len(alphas))
    frames = get_frames_bool(
        data=res,
        iter=iter,
        plotter=plotter,
        fig=fig,
        axes=ax,      # note: single Axes, not a list
    )
    save_frames(frames, path=hydra_out("out.gif"))

    print(f"\n\n{hydra_out('out.gif')} saved\n\n")
    
    if cfg.get('code', None) is not None:
        os.system(f'{cfg.code} {hydra_out("out.gif")}')
    
    

if __name__ == "__main__":
    main()