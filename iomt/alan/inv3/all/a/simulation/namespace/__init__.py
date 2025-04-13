from typing import Callable, Any
from mh.core import StaticClass


class __call__(metaclass=StaticClass):
    def identity(**kwargs):
        return kwargs

    def get(key: str) -> Callable[[Any], Any]:
        d = {
            'id': __call__.identity,
            'identity': __call__.identity,
        }
        return d[key]
    
def get(key):
    d = {
        '__call__': __call__.get
    }
    return d[key]