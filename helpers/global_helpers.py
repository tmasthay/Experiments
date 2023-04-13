from subprocess import check_output as co

#global helpers tyler
class ght:
    def sco(s, split=True):
        u = co(s, shell=True).decode('utf-8')
        if( split ):
            u = u.split('\n')[:-1]
        return u
