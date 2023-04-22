import numpy as np
import argparse
from textwrap import wrap

def generate_sequence(N):
    return np.array(np.round(np.random.random(N)), dtype=int)

def transcribe(x):
    return ['R' if e == 0 else 'B' for e in x]

def pretty_print(trans, c=80, prefix='', suffix=''):
    print('%s%s%s'%(prefix,'\n'.join(wrap(' '.join(trans), width=c)), suffix))

def simulate_prisoners(x, verbose=True, go_slow=False):
    if( verbose or go_slow ):
        pretty_print(
            transcribe(x), 
            prefix=(80*'&' + '\n'), 
            suffix=('\n' + 80*'&')
        )

    parity = np.mod(sum(x[1:]), 2)
    guesses = [parity]
    for i in range(1,len(x)):
        seen = np.mod(sum(x[(i+1):]), 2)
        guesses.append(int(parity != np.mod(sum(x[(i+1):]), 2)))
        prev = parity
        if( guesses[-1] ): parity = int( not parity )
        if( verbose or go_slow ):
            msg = '(true=%s, seen=%s) ::: (prev=%d,seen=%d) ::: says=%s -> updates_parity=%d'%(
                transcribe([x[i]])[0],
                ' '.join(transcribe(x[(i+1):])),
                prev, 
                seen,
                transcribe([guesses[-1]])[0],
                parity
            )
            if( go_slow ):
                input(msg)
            else:
                print(msg)
    return np.array(guesses, dtype=int)

def print_simulation(x, guesses):
    c = 80
    prefix = '\n' + 'REF\n' + c*'*' + '\n'
    suffix = '\n' + c*'*'
    pretty_print(transcribe(x), c, prefix, suffix)
    pretty_print(transcribe(guesses), c, prefix.replace('REF', 'EXP'), suffix)
    check = chr(0x2705)
    cross = chr(0x274C)
    res = [check if a == b else cross for (a,b) in zip(x,guesses)]
    pretty_print(res, c, prefix.replace('REF','COMPARE'), suffix)

def go():
    parser = argparse.ArgumentParser(description='Red Hat Blue Hat')
    parser.add_argument(
        '--N', 
        default=5, 
        type=int,
        help='Number of people in line'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbosity output'
    )
    parser.add_argument(
        '--slow',
        action='store_true',
        help='Show output slowly'
    )
    args = parser.parse_args()
    x = generate_sequence(args.N)
    guesses = simulate_prisoners(x, args.verbose, args.slow)
    print_simulation(x,guesses)

if( __name__ == "__main__" ):
    go()
