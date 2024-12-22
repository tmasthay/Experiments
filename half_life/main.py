import hydra
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig


@hydra.main(config_path="cfg", config_name="cfg", version_base=None)
def main(c: DictConfig) -> None:
    sim_time = c.num_doses * c.reapplication_time
    # t = np.linspace(0, sim_time, int(c.time_upsample * sim_time))

    # y = np.zeros_like(t)
    # for n in range(c.num_doses):
    #     mask = (t >= n * c.reapplication_time) & (t < (n + 1) * c.reapplication_time)

    #     if n == 0:
    #         prev_conc = 0
    #     else:
    #         prev_conc = y[t < n * c.reapplication_time][-1]

    #     y[mask] = (c.dose + prev_conc) * (0.5) ** ((t[mask] - n * c.reapplication_time) / c.half_life)

    # plt.figure(figsize=(10, 6))
    # plt.plot(t, y, label="Medication Concentration")
    # plt.title("Medication Concentration Over Time with Daily Dosing")
    # plt.xlabel("Time (hours)")
    # plt.ylabel("Concentration (Pill Equivalents)")
    # plt.legend()
    # plt.grid()
    # plt.savefig("a.png")
    nt = c.time_upsample * c.num_doses
    y = torch.zeros(nt).to(c.device)
    t = torch.linspace(0, sim_time, nt).to(c.device)
    dose = c.get('dose', 1.0)
    y[0] = dose

    edge = lambda i: c.time_upsample * i
    for n in range(1, c.num_doses):
        curr = edge(n)
        prev = edge(n - 1)
        y[curr] = dose + y[prev] * (0.5) ** (c.reapplication_time / c.half_life)

        # fill in the gaps with exponential decay
        y[(prev + 1) : curr] = y[prev] * (0.5) ** (
            (t[(prev + 1) : curr] - t[prev]) / c.half_life
        )

    if c.cutoff != 0:
        t = t[: -c.cutoff]
        y = y[: -c.cutoff]

    abs_dose = 'dose' in c.keys() and c.abs_dose
    if abs_dose: 
        y_title = f"Concentration Surrogate ({c.dose_units})"
    else: 
        y_title = f"Concentration Surrogate (Dose Equivalents)"
        y = y / dose
    last_sub = y[-2 * c.time_upsample :]

    app_min_steady = last_sub.min()
    app_max_steady = last_sub.max()
    approx_steady_time = t[
        torch.where(y > app_max_steady * (1 - c.steady_tol))[0][0]
    ]

    npi = lambda x: x.cpu().numpy()

    t = npi(t)
    y = npi(y)
    app_min_steady = npi(app_min_steady)
    app_max_steady = npi(app_max_steady)
    approx_steady_time = npi(approx_steady_time)

    base_title = "How Fast to Steady State?\n(Half Life,Reapplication Time) = "
    if c.time_units == 'days':
        t = t / 24
        approx_steady_time = approx_steady_time / 24
        plot_title = (
            f"{base_title}({c.half_life/24:.1f} days,"
            f" {c.reapplication_time/24:.1f} days)"
        )
        x_title = "Time (days)"
    else:
        plot_title = f"{base_title}({c.half_life},{c.reapplication_time})"
        x_title = "Time (hours)"

    plt.plot(t, y)
    plt.axhline(
        app_max_steady,
        color='r',
        linestyle='--',
        label="Approx Steady-State MAX",
    )
    plt.axhline(
        app_min_steady,
        color='g',
        linestyle='--',
        label="Approx Steady-State MIN",
    )
    plt.axvline(
        approx_steady_time,
        color='b',
        linestyle='--',
        label="Approx Steady-State Time",
    )
    plt.title(plot_title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend()
    plt.savefig(c.fig_name)


if __name__ == "__main__":
    main()
