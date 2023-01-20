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
        rows_uni, cols_uni = blocks[0].shape
        rows, cols = rows_uni * rows, cols_uni * cols
    A = np.zeros((rows, cols))
    for (B, indices) in zip(blocks, block_indices):
        if( mode == 'uniform' ):
            row_block, col_block = indices
            row_start, col_start = row_block * rows_uni, col_block * cols_uni
            row_end, col_end = row_start + rows_uni, col_start + cols_uni
        else:
            row_start, col_start = indices
            rows_lcl, cols_lcl = B.shape
            row_end, col_end = row_start + rows_lcl, col_start + cols_lcl
        A[row_start:row_end, col_start:col_end] = B
    return A

#only implements for square matrices for now
def banded_block(blocks, bands, size):
    assert( len(size) == 1 )
    assert(len(blocks) == len(bands))
    for (b, band) in zip(blocks, bands):
        assert(len(b) == len(blocks[0]))
        assert(len(b) - abs(band) > 0)

    nx,ny = blocks.shape
    assert( nx == ny )
    assert( np.mod(size, nx) + np.mod(size, ny) == 0 )
    nbx, nby = int(size / nx), int(size / ny)
    
    banded_matrices = []
    band_indices = []
    for (b, band) in zip(blocks, bands):
        if( band < 0 ):
            band_indices.append([e for e in zip(range(abs(band),nx), range(ny))])
            [banded_matrices.append(b) for i in min(nx-abs(band), ny)]
        else:
            band_indices.append([e for e in zip(range(nx), range(abs(band),ny))])
            [banded_matrices.append(b) for i in min(nx, ny - abs(band))]
    return block_matrix(banded_matrices, bands, rows, cols, 'uniform')

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

if( __name__ == "__main__" ):
    nx = 3
    ny = 6
    x = banded([1,-2,1], [-1,0,1],5)
    y = banded([-1,1], [-1,1], 5)
    z = banded([-1,1], [1,-1], 5)
    A = block_matrix([x,x,x,y,z,y,z],[[0,0],[1,1],[2,2],[0,4],[1,3],[1,5],[2,4]], 3, 6)
    print(A.shape)
    pretty_print(A, blocks=[5,5]) 
