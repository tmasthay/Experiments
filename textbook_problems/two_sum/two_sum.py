import numpy as np
        
def two_sum(x,target):
    idx = np.argsort(x)
    print(idx)
    print(type(idx))
    y = x[idx]
    iL,iR = 0,len(y)-1
    while( iL < iR ):
        s = y[iL] + y[iR]
        if( s < target ):
            iL += 1
        elif( s > target ):
            iR -= 1
        else:
            return idx[iL], idx[iR]
    return ()

if( __name__ == "__main__" ):
    x = np.array([0,0,0,0,0,0,0,0,0,3,4,4,4,5,5,6,6,6,6,9])
    print(two_sum(x, 15))