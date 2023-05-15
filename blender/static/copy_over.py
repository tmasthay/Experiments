from subprocess import check_output as co
import os

u = co('find . -name "*.py"', shell=True).decode('utf-8').split('\n')[:-1]

for uu in u:
    if( 'copy_over' not in uu ):
        os.system('cp %s ../%s'%(uu,uu))
