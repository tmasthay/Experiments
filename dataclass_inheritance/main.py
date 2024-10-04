from abc import ABC, abstractmethod
from dataclasses import dataclass


class A(ABC):
    @abstractmethod
    def __call__(self):
        pass
    
@dataclass(kw_only=True)  
class B(A):
    data: int = 0
    
    def __call__(self):
        print(f'{self.data=}')
    
@dataclass(kw_only=True)
class C(B):
    some_other_data: int = 1
    
    def __call__(self):
        super().__call__()
        print(f'{self.some_other_data=}')
    
c = C(data=10, some_other_data=20)

c()