import math


def getOutputShape(in_width, in_height, k_width, k_height, stride):
    return (
        math.floor((in_width - k_width) / stride) + 1, 
        math.floor((in_height - k_height) / stride) + 1
    )