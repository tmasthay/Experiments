from subprocess import check_output as co
import numpy as np
from scipy.sparse import spdiags, coo_matrix, bmat

def pull_shell(cmd, mode=''):
    txt = co(cmd, shell=True).decode('utf-8')
    if( mode == 'text' ):
        return txt
    elif( mode == 'split' ):
        return txt.split('\n')
    else:
        return txt.split('\n')[:-1]

def kw_def(**kw):
    def helper(key, default):
        return default if key not in kw.keys() else default
    return helper

def pretty_print(A, blocks=[None,None], hor_sep='-', ver_sep='|'):
    rows, cols = A.shape
    R,C = blocks
    for r in range(rows):
        s = ''
        for c in range(cols):
            tmp = '%.1f'%A[r,c]
            if( len(tmp) == 3 ):
                tmp = ' ' + tmp
            s += '%s '%tmp
            if( C != None and np.mod(c,C) == C-1 ):
                s += ' %s '%ver_sep
        print(s)
        if( R != None and np.mod(r,R) == R-1 ):
            print(len(s)*hor_sep)

def symdiag(vals, diags):
    vals = np.array(vals)
    diags = np.array(diags)

    U = np.array([np.diag(v,d) for (v,d) in zip(vals,diags)])
    assert(False not in [uu.shape == U[0].shape for uu in U])
    V = [[''.join([U[k][i][j] for k in range(U.shape[0])]) \
        for i in range(U.shape[1])] \
        for j in range(U.shape[2])]
    V = [['None' if ee == '' else ee for ee in e] for e in V]
    V = ['[' + ','.join(e) + ']' for e in V]
    return '[' + ',\n'.join(V) + ']'

if( __name__ == "__main__" ):
    nx = 5
    ny = 5 
    n = min(nx,ny)
    N = nx * ny
   
    pad = lambda x,n: np.concatenate([x,np.zeros(n)])
    pad1 = lambda v,m,n=0 : pad(v * np.ones(m),n)
 
    A = spdiags([pad1(1,n), pad1(-2,n), pad1(1,n)], [-1,0,1], nx,ny)
    B = spdiags([pad1(-1,n), pad1(1,n)], [-1,1], nx,ny)
    C = spdiags([pad1(-1,n), pad1(1,n)], [1,-1], nx,ny)

    A2 = bmat([[A if i == j else None for j in range(ny)] for i in range(nx)])
    B2 = bmat(eval(symdiag([(n-1)*['B'],(n-1)*['C']], [-1,1])))
   
    D = bmat([[A2,B2],[None,None]])

    pretty_print(D.toarray(), blocks=[nx,ny])
