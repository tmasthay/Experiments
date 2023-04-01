from subprocess import check_output as co
import importlib
import os

def sco(s,split=True):
    try:
        res = co(s,shell=True).decode('utf-8')
    except:
        raise
    if( split ):
        return res.split('\n')[:-1]
    else:
        return res
    
def get_global_folder(fname):
    return fname if '/' not in fname else '/'.join(fname.split('/')[:-1])
    
def get_local_name(s,ext='.py'):
    return s if '/' not in s else s.split('/')[-1].replace(ext, '')
    
def get_subfolders(path, **kw):
    omissions = kw.get('omissions', [])
    local = kw.get('local', True)
    ext = kw.get('ext', '.py')
    depth = kw.get('depth', 1)
    omissions = [path + '/%s'%e if '/' not in e else e for e in omissions]
    try:
        u = sco(r'find %s -type d -depth %d | grep -v "\/[_.]"'%(path, depth))
    except:
        u = []
    if( len(omissions) > 0 ):
        u = [e for e in u if e not in omissions]
    if( local ):
        u = [get_local_name(e, ext) for e in omissions]
    return u

def get_local_modules(path, local=True, ext='.py'):
    res = sco(
        r'find %s -type f -name "*.py" -depth 1 | grep -v "\/[_.]"'%(path)
    )
    if( local ):
        res = [get_local_name(e).replace(ext,'') for e in res]
    return res

def import_local_modules(path, **kw):
    local = kw.get('local', True)
    ext = kw.get('ext', '.py')
    verbose = kw.get('verbose', False)
    local_modules = get_local_modules(path, local=local, ext=ext)
    local_modules = ["'%s'"%e for e in local_modules]
    s = '__all__ = [\n    '
    s += ',\n    '.join(local_modules) + '\n]'
    f = open(path + '/__unit__.py', 'w')
    f.write(s)
    f.close()
    return 0

