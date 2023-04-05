import os
import argparse

def get_color_command(x,y,z, num_spaces=3):
    num_chars = len('%d;%d;%d'%(x,y,z)) + num_spaces
    sp = num_spaces * ' '
    return r"\033[%d;%d;%dm%d;%d;%d\033[0m%s"%(x,y,z,x,y,z,sp), num_chars

def bundle_color_commands(x,y,z, max_length=80):
    cmds = []
    curr = 0
    stmt = ''
    idx = 0
    N = len(x) * len(y) * len(z)
    for yy in y:
        for zz in z:
            for xx in x:
                s, c = get_color_command(xx,yy,zz)
                if( curr + c > max_length ):
                    cmds.append(r'echo "%s"'%stmt)
                    curr = c
                    stmt = s
                else:
                    curr += c
                    stmt += s
                idx += 1
                if( idx == N ):
                    cmds.append(r'echo "%s"'%stmt)
    return cmds

def exec_color_commands(cmds):
    print('Executing %d commands'%len(cmds))
    [os.system(cmd) for cmd in cmds]
    
if( __name__ == "__main__" ):
    parser = argparse.ArgumentParser()
    parser.add_argument('--fore', default=range(30,40), type=eval)
    parser.add_argument('--mode', default=[5], type=eval)    
    parser.add_argument('--colors', default=range(256), type=eval)

    args = parser.parse_args()

    cmds = bundle_color_commands(args.fore, args.mode, args.colors)
    exec_color_commands(cmds) 




