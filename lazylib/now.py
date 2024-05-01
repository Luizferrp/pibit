from datetime import datetime as dt

def now():
    return dt.now().strftime("%d-%m-%Y %H-%M-%S")