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

def inc_calls(func):
    def helper(*args, **kwargs):
        name = str(func).split(' ')[1] + '_calls'
        if( name not in d.keys() ):
            d[name] = 0
        d[name] += 1
        func(*args, **kwargs)
    return helper

def reset_calls(d, verbose=False):
    def decorator(func):
        def helper():
            name = str(func).split(' ')[1]
            d[name] = 0
            if( verbose ):
                print('WARNING: "%s" reset within reset_calls decorator'%name)
        return helper
    return decorator

if( __name__ == "__main__" ):
    tracker1 = dict()
    tracker2 = dict()
 
    inc1 = inc_calls(tracker1)
    inc2 = inc_calls(tracker2)
    
    @inc1
    def foo(*args, **kwargs):
        print('foo called %d time(s)...'%(tracker1['foo']),end='')
        print('Latest call with args="%s", kwargs="%s"'%(str(args), str(kwargs)))

    @inc2
    def bar(a,b):
        print('bar called %d time(s)...'%(tracker2['bar']),end='')
        print('Latest call with a="%s", b="%s"'%(str(a), str(b)))

    @inc1
    def baz():
        print('baz called %d time(s)...'%(tracker1['baz']))

    @inc2
    def boom():
        print('boom called %d times...tracker1="%s"...tracker2="%s"'%(tracker2['boom'], tracker1, tracker2))

    N = 100
    calls = [int(i) for i in np.round(np.random.random(N) * 3)]
    funcs = [foo, bar, baz, boom]
    boom()
    for c in calls:
        if( c == 0 ): funcs[0](1,2,key1='yes',key2=(lambda x : x)) 
        elif( c == 1 ): funcs[1](np.random.random(), np.random.random())
        else: funcs[c]() 
