def get_four():
    return [1,2,3,4]

def get_eight():
    return [1,2,3,4,5,6,7,8]

def get_two():
    return [1,2]

def get(key):
    d = {
        'four': get_four(),
        'eight': get_eight(),
        'two': get_two() 
    }
    return d[key]