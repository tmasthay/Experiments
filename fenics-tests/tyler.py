from subprocess import check_output as co
import numpy as np

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

def banded(vals, bands, d):
    assert( len(vals) == len(bands) )
    assert( len(vals) > 0 )
    for i in range(len(vals)):
        if( type(vals[i]) in [float, int] ):
            vals[i] = vals[i] * np.ones(d-abs(bands[i]))
        elif( len(vals[i]) != d - abs(bands[i]) ):
            raise ValueError('Wrong input: (expect_size, receieved_size) = (%d,%d)'%(d-abs(bands[i]), len(vals[i])))

    A = np.zeros((d,d))
    for (v,b) in zip(vals, bands):
        A = A + np.diag(v, b)
    return A

def block_matrix(blocks, block_indices, rows, cols, mode='uniform'):
    assert(len(blocks) == len(block_indices))
    if( mode == 'uniform' ):
        for b in blocks:
            assert(blocks[0].shape == b.shape)
    A = np.zeros((rows, cols))
    for (B, indices) in zip(blocks, block_indices):
        row_start = indices[0]
        col_start = indices[1]
        rows_lcl, cols_lcl = B.shape
        row_end = row_start + rows_lcl
        col_end = col_start + cols_lcl
        A[row_start:row_end, col_start_col_end] = B
    return A

    
