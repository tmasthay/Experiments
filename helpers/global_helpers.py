from subprocess import check_output as co
import os

#global helpers tyler
class ght:
    def sco(s, split=True):
        u = co(s, shell=True).decode('utf-8')
        if( split ):
            u = u.split('\n')[:-1]
        return u

    def conda_include_everything():
        inc_paths = ':'.join(ght.sco('find $CONDA_PREFIX/include -type d'))
        c_path = os.environ("C_INCLUDE_PATH")
        cmd = "echo 'export C_INCLUDE_PATH=%s:%s'"%(c_path, inc_paths)
        cmd += ' | pbcopy'
        os.system(cmd) 
        
