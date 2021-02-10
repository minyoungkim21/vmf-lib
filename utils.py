import os


realmin = 1e-10


def norm(input, p=2, dim=0, eps=1e-12):
    return input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


def mkdirs(path):    
    if not os.path.exists(path):
        os.makedirs(path)

