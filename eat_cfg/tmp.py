from mh.core import exec_imports, DotDict
from misfit_toys.utils import resolve

u = DotDict(
    dict(
        a='^^torch.nn',
        b='^^torch.nn.functional',
        c='self.a.Linear',
        d='^^torch.nn|Linear',
        e='^^cwd|helpers|my_func'
    )
)
print(f'After construction: {u}')

v = exec_imports(u)

print(
    f'u after calling exec_imports and assigning to v: {u}\n\nSide effects! -->'
    f' {(u==v)=}'
)

w = resolve(v, relax=False)

print(
    f'u after call resolve(v,relax=False) and assigning to w\n\n{u=}\n\nSide'
    f' effects! --> {(u==w)=}'
)

print('\nCalling u.e() now!\n')
u.e()
