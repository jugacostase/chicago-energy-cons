#hey
def generate_filename(year, month=None, day=None, day_type=None):
    if (day == None) & (month !=None):
        if month < 10:
            month = f'0{month}'
        filename = f'{year}{month}'
    elif day != None:
        if day < 10:
            day = f'0{day}'
        filename = f'{year}{month}{day}'
    else:
        filename = f'{year}'

    if day_type != None:
        filename = f'{filename}_{day_type}'

    return filename