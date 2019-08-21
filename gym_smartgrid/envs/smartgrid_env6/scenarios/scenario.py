
def init_vre_easy():
    generators = [_constant_generator(5.), _constant_generator(10.)]
    return generators

def init_load_easy():
    generators = [_constant_generator(-5.),
                  _constant_generator(-10.),
                  _constant_generator(-20.)]
    return generators

def init_soc_easy(soc_max):
    return soc_max

def _constant_generator(value):
    while True:
        yield value



