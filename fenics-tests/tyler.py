from subprocess import check_output as co
import numpy as np
from scipy.sparse import spdiags, coo_matrix, bmat
import time

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
# symbolic diagonal
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

def rtgs(**kw):
    rick_t0 = kw.get('rick_t0', 0.0)
    rick_sig = kw.get('rick_sig', 1.0)

    gx_mu = kw.get('gx_mu', 1.0)
    gx_sig = kw.get('gx_sig', 1.0)

    gy_mu = kw.get('gy_mu', 1.0)
    gy_sig = kw.get('gy_sig', 1.0)

    amp = kw.get('amp', 1.0)

    def helper(x,y,t):
        tau = (t-rick_t0) / rick_sig
        x_tilde = (x - gx_mu) / gx_sig
        y_tilde = (y - gy_mu) / gy_sig
        return amp * (1 - tau**2) * np.exp(-tau**2 - x_tilde**2 - y_tilde**2)
    return helper

def damp1(**kw):
    a = kw.get('a', -1.0)
    b = kw.get('b', 1.0)
    p = kw.get('p', 0.1)
    s = kw.get('s', 10.0)

    def helper(x):
        c = (1-p) * a + p * b
        d = p * a + (1-p) * b

        if( x < c ):
            return np.exp(-(c-x) * s)
        elif( x > d ):
            return np.exp(-(x-d) * s)
        else:
            return 1.0
    return helper

def damp2(**kw):
    ax = kw.get('ax', -1.0)
    bx = kw.get('bx', 1.0)
    px = kw.get('px', 0.3)
    sx = kw.get('sx', 1.0)

    ay = kw.get('ay', -1.0)
    by = kw.get('by', 1.0)
    py = kw.get('py', 0.3)
    sy = kw.get('sy', 1.0)

    f1 = damp1(a=ax, b=bx, p=px, s=sx)
    f2 = damp1(a=ay, b=by, p=py, s=sy)

    def helper(x,y):
        return f1(x) * f2(y)
    return helper

def read_input_dict(file_name):
    l = open(file_name).read().split('\n')
    while( '' in l ):
        l.remove('')
    u = [ll.split('=') for ll in l]
    print(u)
    return eval('{' + '\n'.join(["'%s': %s,"%(ll[0],ll[1]) for ll in u]) + '}')
    

class dec_nsp:
    env = dict()

    def dec(func):
        def helper(*args, **kwargs):
            name = str(func)
            if( name not in dec_nsp.env.keys() ):
                dec_nsp.env[name] = {'calls': 1}
                print(dec_nsp.env)
            else:
                dec_nsp.env[name]['calls'] += 1
                print(dec_nsp.env)
            func(*args, **kwargs)
        return helper

    def dec_obj(obj='bound', verbose=False, decorator_action=(lambda f,d,t,*a,**k: \
            f(*a,*k))):
        bound = ['bound', 'class']
        def dec(func):
            def helper(*args, **kwargs):
                if( len(args) == 0 and obj in bound ):
                    raise ValueError('dec_obj is meant to decorate %s'%(
                        'class functions with signatures of the form %s'%(
                        'f(self, *args, **kwargs)...your function is %s'%(
                        'of the form f(**kwargs), which is independent %s'%(
                        'of the instance of the class, i.e., static\n')))))
                elif( obj not in bound ):
                    obj_str = 'unbound'
                else:
                    obj_str = str(args[0])
                f_str = str(func)
                d = dec_nsp.env
                if( obj_str not in d.keys() ):
                    d[obj_str] = dict()
                if( f_str not in d[obj_str] ):
                    d[obj_str][f_str] = dict()
                t = time.time()
                out_val = func(*args, **kwargs) 
                t = time.time() - t
                decorator_action(out_val, d[obj_str][f_str], t, \
                    *args, **kwargs)
                if( verbose ):
                    print('(obj,func,call_time)=(%s,%s,%f)'%(obj_str,
                        f_str.split(' ')[1], t))
                return out_val
            return helper
        return dec

    def inc_obj(obj='bound', verbose=False):
        def decorator_action(f_out, d, t, *args, **kwargs):
            d['calls'] = 1 if 'calls' not in d.keys() else d['calls'] + 1
        return dec_nsp.dec_obj(obj, verbose, decorator_action)

    def inc_timer(obj='bound', verbose=False):
        def decorator_action(f_out, d, t, *args, **kwargs):
            if( 'calls' not in d.keys() ): 
                d['calls'] = 1
                d['time'] = [t]
            else:
                d['calls'] += 1
                d['time'].append(t)
        return dec_nsp.dec_obj(obj, verbose, decorator_action)
    
    def get_meta(obj):
        s = str(obj)
        if( s in dec_nsp.env.keys() ):
            return dec_nsp.env[s]
        else:
            return None

    def report_meta(obj):
         s = str(obj)
         print('METADATA FOR %s'%s)
         print(80*'*')
         print(str(dec_nsp.get_meta(obj)))
         print(80*'*' + '\n')

if( __name__ == "__main__" ):
    class MyClass:
        def __init__(self, xx=0):
            self.x = xx
            self.inc = dec_nsp.dec_obj(str(self))
    
        @dec_nsp.inc_obj('bound')
        def set_x(self, xx):
            self.x = xx
    
        @dec_nsp.inc_obj('bound')
        def static_method_treated_nonstatically():
             print('Should get error raised before this statement hits.')
    
        @dec_nsp.inc_obj('unbound')
        def static_method_correct():
             print('Called static method...no error raised')
    
        @dec_nsp.inc_obj('bound')
        def static_method_unavoidable_bug(arg1_not_self):
             print('Here we have treated our first argument like %s'%(
                 'it is self but is not truly self...this is on the user, %s'%(
                 'not me')))
    
        @dec_nsp.inc_obj('unbound')
        def static_method_bug_workaround(arg1_not_self):
             print('Here we have treated first arg like we should')

    u = MyClass()
    v = MyClass()

    u.set_x(1)
    u.set_x(2)
    v.set_x(3)

    try:
        MyClass.static_method_treated_nonstatically()
    except ValueError as e:
        print(e)

    MyClass.static_method_correct()
    MyClass.static_method_unavoidable_bug(1.0)
    MyClass.static_method_unavoidable_bug(2.0)
    MyClass.static_method_bug_workaround(3.0) 

    dec_nsp.report_meta(u)
    dec_nsp.report_meta(v)

    print("""
        Now let's check the unbound static methods. We should see counts
        for static_method_correct and static_method_bug_workaround but not
        for static_method_unavoidable_bug
        and static_method_treated_nonstatically""")
    dec_nsp.report_meta('unbound')

    print("""
       Now checking the full dictionary, we do see that info for the
       incorrect static treatment is there but it is not formatted cleanly
       """)
    print(dec_nsp.env)
