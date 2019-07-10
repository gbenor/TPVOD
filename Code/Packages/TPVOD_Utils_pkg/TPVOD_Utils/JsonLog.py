import json
import datetime

data = {}
filename =""
Creation_time = ""


def json_save():
    json.dump(data, open(filename, 'w'))

def add_to_json (k,v):
    data[k] = v
    json_save()

def set_filename (f):
    global filename
    global data
    global Creation_time

    filename = str(f)
    data = {}
    Creation_time = str(datetime.datetime.utcnow())
    add_to_json('Creation_time', Creation_time)

def get_creation_time ():
    return Creation_time

def json_print ():
    print (json.dumps(data))
