import random


def random_string(num_characters=10):
    result = "".join(random.choice("abcdef0123456789") for i in range(num_characters))
    return result
