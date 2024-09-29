from dotmap import DotMap
import hydra
from omegaconf import DictConfig, OmegaConf

def calculate_tax(income: float, tax_brackets: list, standard_deduction: float) -> float:
    taxable_income = max(0, income - standard_deduction)
    total_tax = 0

    for bracket in tax_brackets:
        lower_bound, upper_bound, rate = bracket
        if taxable_income > lower_bound:
            taxable_amount = min(taxable_income, upper_bound) - lower_bound
            total_tax += taxable_amount * rate
        else:
            break
    
    return total_tax

def preprocess_cfg(cfg: DictConfig) -> DotMap:
    return DotMap(OmegaConf.to_container(cfg, resolve=True))

@hydra.main(config_path='cfg', config_name='cfg', version_base=None)
def main(cfg: DictConfig) -> None:
    c = preprocess_cfg(cfg)

    
    tax_info = c[c.mode]
    standard_deduction = tax_info.standard_deduction
    tax_brackets = tax_info.brackets
    
    tax = calculate_tax(c.income, tax_brackets, standard_deduction)
    
    
    
    print(f"Tax for {c.mode} with income ${c.income:.2f} is: ${tax:.2f}")
    

if __name__ == "__main__":
    main()
