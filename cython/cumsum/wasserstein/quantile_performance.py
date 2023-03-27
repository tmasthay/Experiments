from scipy.stats.sampling import NumericalInverseHermite
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.stats.mstats import mquantiles
import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import accumulate
import os
from quantile import *
import seaborn as sns
from quantile_cython import quantile as quantile_cython
import copy
from smart_quantile_cython import *

class Dummy:
    def __init__(self, pdf_arg, cdf_arg):
        self.pdf = pdf_arg
        self.cdf = cdf_arg

def create_call(d, k, f, args, kw, prebuild=False):
    d.update({k : {'f': f, 'args': args, 'kw': kw, 'prebuild': prebuild}})

def tme(f, *args, **kw):
    t = time.time()
    v = f(*args, **kw)
    return v, time.time() - t

def create_calls_dict(d, **kw):
    #mandatory keyword args
    x = kw['x']
    f = kw['f']

    #Optional keyword arguments
    div_tol = kw.get('div_tol', 1e-16)
    tail_tol = kw.get('tail_tol', 1e-30)

    #compute other vars based on keyword args
    dx = x[1] - x[0]
    ox = x[0]
    F = cumulative_trapezoid(f, dx=dx, initial=0.0)

    calls = dict()
    cc = lambda k,f,a,kw,pb: create_call(d,k,f,a,kw,pb)
    cc(
        'Hermite',
        NumericalInverseHermite, 
        [Dummy(interp1d(x, f, kind='cubic'), interp1d(x, F, kind='cubic'))],
        {'domain': (x[0],x[-1]), 'order': 3, 'u_resolution': 1e-5},
        True
    )
    cc(
        'Manual', 
        quantile, 
        [dx,ox], 
        {'domain': (x[0], x[-1]), 'order': 3, 'u_resolution': 1e-5}
    )

    for k,v in d.items():
        if( k == 'Hermite' ):
            d[k].update({'eval': v['f'](*v['args'], **v['kw']).ppf})
        else:
            d[k].update({
                'eval': lambda p : v['f'](p,F,*v['args'], **v['kw'])
                }
            )
    return calls

def time_calls(d, p, verbose=False):
    res = dict()
    for k,v in d.items():
        if( verbose ): print(k)
        t = time.time()
        val = v['eval'](p)
        exec_time = time.time() - t
        res[k].update({'val': val, 'exec_time': exec_time})
    return res

def create_data(**kw):
    #get optional kwargs
    a = kw.get('a', -5)
    b = kw.get('b', 5)
    N_base = int(kw.get('N', 1000))
    verbose = kw.get('verbose', True)
    mu = kw.get('mu', 0.0)
    sig = kw.get('sig', 1.0)

    #evaluate directly derived quantities
    x = np.linspace(a,b,N_base)
    dist = norm(loc=mu, scale=sig)
    f = dist.pdf(x)
    F = dist.cdf(x)

    #build various quantile functions
    calls = create_calls_dict(x=x, f=f)

    #compute results
    res = time_calls(calls, F, verbose)

def make_plots(
        N_vals, 
        eval_times,
        errors,
        case_names,
        error_names,
        x_vals,
        cdfs,
        p_vals,
        quantiles,
        scale='log',
        eta=0.0
):
    folder = 'quantile_plots'
    os.system('mkdir -p %s'%folder)

    num_cases = len(eval_times)
    num_norms = len(error_names)

    palette = sns.color_palette('colorblind', 2 * num_cases)
    colors = [rgb for rgb in palette.as_hex()]
    lsty = ['--', '-.', ':']

    get = lambda arr : lambda idx : arr[np.mod(idx, len(arr))]
    get_color = get(colors)
    get_lsty = get(lsty)

    plt.rc('text', usetex=True)
    for i in range(len(N_vals)):
        plt.plot(
            x_vals[i],
            cdfs[i],
            color=get_color(i),
            linestyle=get_lsty(i),
            label=str(len(N_vals))
        )
    plt.title("CDF at different resolutions")
    plt.legend()
    plt.savefig('%s/cdf.pdf'%folder)
    plt.clf()

    for i in range(len(N_vals)):
        # plt.subplot(2,1,1)
        plt.plot(cdfs[i], x_vals[i], label='Reference', color=get_color(-1))
        for j in range(num_cases):
            plt.plot(
                p_vals[i],
                quantiles[i][j],
                color=get_color(j),
                linestyle=get_lsty(j),
                label=case_names[j]
            )
        # plt.title("Quantile: %d samples"%N_vals[i])
        # plt.legend()
        # plt.subplot(2,1,2)
        # plt.plot(
        #     x_vals[i],
        #     cdfs[i],
        #     color=get_color(i),
        #     linestyle=get_lsty(i),
        #     label='CDF'
        # )
        plt.legend()
        plt.savefig('%s/quantiles_%d.pdf'%(folder, N_vals[i]))
        plt.clf()

    for i in range(num_cases):
        plt.plot(
            N_vals,
            eval_times[i], 
            color=get_color(i), 
            linestyle=get_lsty(i),
            label=case_names[i]
        )
    plt.xlabel('Problem size')
    plt.ylabel('Time (s)')
    plt.yscale(scale)
    plt.legend()
    plt.title('Noise level: %.4e'%eta)
    plt.savefig('%s/runtimes.pdf'%folder)
    plt.clf()

    for j in range(num_norms):
        for i in range(num_cases):      
            plt.plot(
                N_vals, 
                errors[i,j,:], 
                color=get_color(i), 
                label=r'%s: %s'%(error_names[j], case_names[i]),
                linestyle=get_lsty(i)
            )
        plt.xlabel('Problem size')
        plt.ylabel('Error')
        plt.yscale(scale)
        plt.legend()
        plt.title('Noise level: %.4e'%eta)
        plt.savefig('%s/err_%d.pdf'%(folder, j))
        plt.clf()

def go():
    a = -5
    b = 5
    mu = 0.0
    sig = 1.0
    dist = norm(loc=mu, scale=sig)

    N_base = int(1e3)
    x_tmp = np.linspace(a,b,N_base)

            
    dummy = Dummy(
        interp1d(x_tmp, dist.pdf(x_tmp), kind='cubic'),
        interp1d(x_tmp, dist.cdf(x_tmp), kind='cubic'))
    
    t = time.time()
    q, setup_time = \
        tme(
            NumericalInverseHermite, 
            dummy, 
            domain=(a,b), 
            order=3, 
            u_resolution=1e-5
        )
    setup_time = time.time() - t

    N_vals = np.array([2**e for e in range(6,15)])
    noise_level = 0.0
    noise_level = 500.0
    case_names = ['ScipyHermite', 'ScipyQuantile', 'PurePython', 'Cython']
    non_np_modes = len(case_names)
    np_modes = [
        'inverted_cdf',
        'averaged_inverted_cdf',
        'closest_observation',
        'interpolated_inverted_cdf',
        'hazen',
        'weibull',
        'linear',
        'median_unbiased',
        'normal_unbiased'
    ]
    np_modes = []
    [case_names.append(e) for e in np_modes]
    error_names = [r'$\ell_2$', r'$\ell_{\infty}$']
    num_cases = len(case_names)
    eval_times = np.empty((num_cases,len(N_vals)))
    errors = np.empty((num_cases, num_cases, len(N_vals)))
    res = [[] for i in range(num_cases)]
    cdfs = []
    x_vals = []
    p_vals = []
    q_vals = []

    eps = 1e-5
    u = time.time()
    for (i,N) in enumerate(N_vals):
        elapsed = time.time() - u
        computed = max(sum(N_vals[:i]),1)
        remainder = sum(N_vals[i:])
        avg = elapsed / computed
        remaining = avg * remainder
        print('Iteration %d of %d...elapsed=%f...etr: %.2f'%(
            i+1,
            len(N_vals),
            time.time() - u,
            remaining
            )
        )

        x = np.linspace(a,b,N)
        p = np.linspace(eps, 1-eps, N)
        dx = x[1] - x[0]
        if( noise_level == 0.0 ):
            Y = dist.cdf(x)
            ref = dist.ppf(p)
        elif( 1 == 0 ):
            tmp_pdf = noise_level * np.random.random(len(x))
            tmp_pdf = np.abs(tmp_pdf - noise_level/2)
            mid = len(tmp_pdf) // 2
            delta = mid // 2 
            tmp_pdf[(mid-delta):(mid+delta)] = 0.0
            Y = cumulative_trapezoid(tmp_pdf, dx=dx, initial=0.0)
            Y /= Y[-1]
        else:
            num_samples = min(3, len(x))
            the_pdf = np.zeros(len(x))
            idx = np.random.choice(len(x), num_samples, replace=False)
            flat_no = 5
            for i_idx in range(len(idx)):
                val = np.random.random()
                for i_flat in range(flat_no):
                    curr = idx[i_idx] + i_flat
                    if( curr < len(the_pdf) ):
                        the_pdf[curr] = val
            Y = cumulative_trapezoid(the_pdf, dx=dx, initial=0.0)
            Y /= Y[-1]
            the_pdf /= Y[-1]

            dummy = Dummy(
                interp1d(x, the_pdf, kind='cubic'),
                interp1d(x, Y, kind='cubic'))

            try:            
                q = NumericalInverseHermite( 
                    dummy, 
                    domain=(a,b), 
                    order=3, 
                    u_resolution=1e-5
                )
            except:
                q = dummy
                q.ppf = lambda x : x


        res[0], eval_times[0,i] = tme(q.ppf, p)
        res[1], eval_times[1,i] = tme(mquantiles, Y, p)
        res[2], eval_times[2,i] = tme(quantile, Y, p, dx, a)
        #res[3], eval_times[3,i] = tme(quantile_cython, Y, p, dx, a)
        res[3], eval_times[3,i] = tme(quantile, Y, p, dx, a)
        for (c,e) in enumerate(np_modes):
            cc = c + non_np_modes
            res[cc], eval_times[cc,i] = tme(
                np.quantile,
                Y,
                p,
                method=e
            )
        if( noise_level > 0 ):
            ref_idx = 0
            ref = res[ref_idx]

        for j in range(num_cases):
            diff = ref - res[j]
            errors[j,0,i] = np.linalg.norm(diff) / N
            errors[j,1,i] = np.max(diff)
        x_vals.append(x)
        cdfs.append(Y)
        p_vals.append(p)
        q_vals.append(copy.copy(res))

    scale = 'linear'
    make_plots(
        N_vals, 
        eval_times, 
        errors, 
        case_names, 
        error_names,
        x_vals,
        cdfs,
        p_vals,
        q_vals, 
        scale,
        noise_level
    )

def get_flat_subintervals2(x, tol=0.0):
    idx = np.where(x <= tol)[0]
    flat_int = []
    if( len(idx) > 1 ):
        start = idx[0]
        prev = idx[0]
        runner = 1
        for i in range(1, len(idx)):
            curr = idx[i]
            if( curr == prev + 1 ):
                runner += 1
            else:
                if( runner > 2 ):
                    flat_int.append((start+1, prev-1))
                start = curr
            prev = curr
        if( len(flat_int) > 0 ):
            return flat_int
        else:
            return [(np.inf, np.inf)]
    else:
        return [(np.inf, np.inf)]


def smart_quantile2(x, pdf, cdf, p, tol=0.0):
    flat_int = get_flat_subintervals2(pdf)
    assert( np.min(pdf) >= 0.0 )
    assert( np.abs( np.max(cdf) - 1.0 ) < 1e-8 )
    assert( np.abs( np.min(cdf) ) < 1e-8 )
    assert( np.all([cdf[i] >= cdf[i-1] for i in range(1,len(cdf))]) )
    nsubs = len(flat_int)
    sidx = 0
    N = len(x)
    P = len(p)
    q = np.empty(P)
    q[0] = x[0]
    q[-1] = x[-1]
    i = 1
    i_x = 0
    flat_int = [(np.inf, np.inf)]
    for i in range(1,P-1):
        if( i_x == N - 1 ):
            q[i:] = x[1]
            break
        while( cdf[i_x] > p[i] or p[i] > cdf[i_x+1] ):
            i_x += 1
            csub = flat_int[sidx]
            if( csub[0] <= i_x and i_x <= csub[1] ):
                print('%d @@@ %d @@@ %d'%(csub[0], i_x, csub[1]))
                i_x = csub[1] + 1
                if( sidx < nsubs - 1 ):
                    sidx += 1
            if( i_x == N - 1 ):
                q[i:] = x[1]
                break
        if( i_x == N - 1 ):
            q[i:] = x[1]
            break
        delta = cdf[i_x+1] - cdf[i_x]
        if( delta > 0 ):
            alpha = (p[i] - cdf[i_x]) / delta
            q[i] = (1.0 - alpha) * x[i_x] + alpha * x[i_x+1]
        else:
            q[i] = x[i_x]
    return q

if( __name__ == "__main__" ):
    N = 10
    x = np.linspace(-5,5,N)
    dx = x[1]-x[0]

    num_intervals = N // 10
    fn = 3
    idx = np.random.choice(N, num_intervals, replace=False)
    u = np.zeros(N)
    touched = []
    for i in idx:
        val = np.random.random()
        for j in range(fn):
            curr = i + j
            if( curr not in touched and curr < N ):
                u[curr] = val
                touched.append(curr)
    U = cumulative_trapezoid(u, dx=dx, initial=0.0)
    U /= U[-1]

    p = np.linspace(0,1,N+1)
    tol = 0.0

    num_trials = 100
    py_time = 0
    cy_time = 0
    for trial in range(num_trials):
        # idx = np.random.choice(N, num_intervals, replace=False)
        # u = np.zeros(N)
        # touched = []
        # for i in idx:
        #     val = np.random.random()
        #     for j in range(fn):
        #         curr = i + j
        #         if( curr not in touched and curr < N ):
        #             u[curr] = val
        #             touched.append(curr)
        u = np.random.random(N)
        U = cumulative_trapezoid(u, dx=dx, initial=0.0)
        U /= U[-1]
        t = time.time()
        q_py = smart_quantile2(x, u, U, p, tol)
        py_time += time.time() - t
        t = time.time() 
        q_cy = smart_quantile(x, u, U, p, tol)
        cy_time += time.time() - t


    plt.subplot(2,1,1)
    plt.plot(x, U, label='CDF')
    plt.plot(x, u, label='PDF')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(p, q_py, label='Python', color='blue')
    plt.plot(p, q_cy, label='Cython', color='green', linestyle=':')
    plt.plot(U, x, label='Reference', linestyle='-.', color='red')
    plt.legend()
    plt.savefig('quantile_plots/AAA.pdf')

    print('Python: %.8e'%(py_time / num_trials))
    print('Cython: %.8e'%(cy_time / num_trials))

                


