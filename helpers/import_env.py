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

def superpkgs(path):
    superpackages = []
    curr = path.split('/')[:-1]
    found = True
    while( len(curr) > 1 ):
        cmd = 'find %s -name __init__.py -depth 1'%('/'.join(curr))
        found = len(sco(cmd)) > 0
        if( found ):
            superpackages.insert(0,curr[-1])
            curr = curr[:-1]
        else:
            break
    return superpackages

def get_local_modules(path, local=True, ext='.py'):
    res = sco(
        r'find %s -type f -name "*.py" -depth 1 | grep -v "\/[_.]"'%(path)
    )
    if( local ):
        res = [get_local_name(e).replace(ext,'') for e in res]
    return res

def import_local_modules(path, local=True, ext='.py', super_module=True):
    input(path)
    local_modules = get_local_modules(path, local=local, ext=ext)
    module_name = get_local_name(path)
    input(module_name)
    input(local_modules)
    for e in local_modules:
        # importlib.import_module(e)
        exec('from .%s import *'%e)
    return local_modules

def import_submodules(path, **kw):
    subfolders = get_subfolders(path, **kw)
    if( len(subfolders) == 0 ):
        return None
    else:
        for folder in subfolders:
            # importlib.module(folder)
            exec('import %s'%folder)
        return subfolders
    
def import_dependencies(path, **kw):
    filename = kw.get('filename', True)
    if( filename ):
        path = get_global_folder(path)
    all_dummy = import_local_modules(path)
    submodules = import_submodules(path, **kw)
    [all_dummy.append(e) for e in submodules if e != None]
    return all_dummy



