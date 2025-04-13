def happy_birthday():
    return "Happy Birthday"

def hello_world():
    return "Hello, World!"

def get(key):
    d = {
        'happy_birthday': happy_birthday(),
        'hello_world': hello_world()
    }
    return d[key]