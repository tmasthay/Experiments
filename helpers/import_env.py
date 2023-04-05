from subprocess import check_output as co
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
    
def get_local_name(s,ext='.py'):
    return s if '/' not in s else s.split('/')[-1].replace(ext, '')
    
def get_subfolders(path, **kw):
    omissions = kw.get('omissions', [])
    local = kw.get('local', True)
    ext = kw.get('ext', '.py')
    depth = kw.get('depth', 1)
    omissions = [path + '/%s'%e if '/' not in e else e for e in omissions]
    try:
        cmd = r'find %s -type d -depth %d | grep -v "\/[_.]"'%(path, depth)
        u = sco(cmd)
        u = [e for e in u \
            if not e.startswith('__') \
                and not e.startswith('.')
        ]
    except:
        u = []
    if( len(omissions) > 0 ):
        u = [e for e in u if e not in omissions]
    if( local ):
        u = [get_local_name(e, ext) for e in u]
    return u

def get_local_modules(path, **kw):
    local = kw.get('local', True)
    ext = kw.get('ext', '.py')
    res = sco(
        r'find %s -type f -name "*.py" -depth 1'%(path)
    )
    res = [e for e in res if not (e.startswith('.') or e.startswith('_'))]
    if( local ):
        res = [get_local_name(e).replace(ext,'') for e in res]
    return res

def init_modules(path, **kw):
    root = kw.get('root', False)
    if( root ):
        local_modules = []
    else:
        local_modules = get_local_modules(path, **kw)
    subfolders = get_subfolders(path, **kw)
    [local_modules.append(e) for e in subfolders]
    s = '__all__ = [\n'
    for e in local_modules:
        s += '    "%s",\n'%e
    s = s[:-2]
    s += '\n]\n'
    s += 'from . import *'
    if( not root ):
        filename = path + '/__init__.py'
        with open(filename, 'w') as f: 
            f.write(s)

    global_subfolders = ['%s/%s'%(path,e) for e in subfolders]
    kw['root'] = False
    for e in global_subfolders:
        init_modules(e, **kw)

def run_make_files(omissions=[]):
    ref_path = os.getcwd()
    omissions = [ \
        e \
        if e.startswith('/') \
        else '%s/%s'%(ref_path, e) \
        for e in omissions \
    ]
    make_files = sco('find %s -name "Makefile"'%ref_path, True)
    make_dir = [e.replace('/Makefile','') for e in make_files]
    make_dir = [e for e in make_dir if e in omissions]
    print(make_dir)

