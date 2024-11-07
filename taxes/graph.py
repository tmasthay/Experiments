from dotmap import DotMap
import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
import torch


def calculate_tax(
    income: float, tax_brackets: list, standard_deduction: float
) -> float:
    taxable_income = max(0, income - standard_deduction)
    total_tax = 0

    for bracket in tax_brackets:
        lower_bound, upper_bound, rate = bracket
        if taxable_income >= upper_bound:
            total_tax += (upper_bound - lower_bound) * rate
        elif lower_bound <= taxable_income < upper_bound:
            total_tax += (taxable_income - lower_bound) * rate

    return total_tax


def get_effective_tax_rate(
    tax_brackets: list,
    standard_deduction: float,
    N: int = 100,
    max_income=None,
    highlighted_income=None,
) -> torch.Tensor:
    if max_income is None:
        # default to 10x the highest bracket
        max_income = tax_brackets[-1][0] * 2.0

    incomes = torch.linspace(0, max_income, N)
    taxes = torch.zeros_like(incomes)

    for i in range(N):
        taxes[i] = calculate_tax(incomes[i], tax_brackets, standard_deduction)

    if highlighted_income is not None:
        highlighted_etr = (
            calculate_tax(highlighted_income, tax_brackets, standard_deduction)
            / highlighted_income
            * 100.0
        )
    else:
        highlighted_etr = None

    return incomes, taxes / incomes * 100.0, highlighted_etr


def unit_to_string(unit):
    d = {
        1: 'Dollars',
        1000: 'Thousands of dollars',
        100000: 'Hundreds of thousands of dollars',
        1000000: 'Millions of dollars',
    }
    return d[unit]


def plot_effective_taxes(
    *,
    incomes,
    effective_tax_rates,
    filename='effective_tax_rates.png',
    highlighted_income=None,
    highlighted_etr=None,
    unit=100000,
):
    unit_str = unit_to_string(unit)
    incomes = incomes / unit
    if highlighted_income is not None:
        highlighted_income = highlighted_income / unit
    plt.plot(incomes, effective_tax_rates)
    plt.xlabel(f'Income ({unit_str})')
    plt.ylabel('Effective Tax Rate (%)')
    plt.title('Effective Tax Rate vs Income')
    if highlighted_income is not None:
        plt.scatter(highlighted_income, highlighted_etr, color='red')
    plt.grid(True)
    plt.savefig(filename)


def preprocess_cfg(cfg: DictConfig) -> DotMap:
    return DotMap(OmegaConf.to_container(cfg, resolve=True))


@hydra.main(config_path='cfg', config_name='graph', version_base=None)
def main(cfg: DictConfig) -> None:
    c = preprocess_cfg(cfg)

    tax_info = c[c.mode]
    standard_deduction = tax_info.standard_deduction
    tax_brackets = tax_info.brackets

    incomes, effective_tax_rates, highlighted_etr = get_effective_tax_rate(
        standard_deduction=standard_deduction,
        tax_brackets=tax_brackets,
        N=c.num_samples,
        max_income=c.max_income,
        highlighted_income=c.ref_income,
    )
    plot_effective_taxes(
        incomes=incomes,
        effective_tax_rates=effective_tax_rates,
        filename=c.filename,
        highlighted_income=c.ref_income,
        highlighted_etr=highlighted_etr,
        unit=c.unit,
    )


if __name__ == "__main__":
    main()
