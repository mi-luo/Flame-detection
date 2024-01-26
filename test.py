import torch
import torch.nn as nn

def drop_space(x):
    b, w, h, c = x.size()
    mask = torch.ones_like(x)
    top = torch.randint(2, [1]).item()
    left = torch.randint(2, [1]).item()
    right = torch.randint(2, [1]).item()
    bottom = torch.randint(2, [1]).item()

    if top:
        c_choice = torch.randint(c, (c // 4,))
        mask[:, :, 0, c_choice] = 0
    if left:
        c_choice = torch.randint(c, (c // 4,))
        mask[:, 0, :, c_choice] = 0
    if right:
        c_choice = torch.randint(c, (c // 4,))
        mask[:, :, -1, c_choice] = 0
    if bottom:
        c_choice = torch.randint(c, (c // 4,))
        mask[:, -1, :, c_choice] = 0
    x = mask * x
    return x

if __name__ == "__main__":
    for i in range(1000):
        if i % 100 == 0:
            print(i/1000)
        batch_image = torch.randn((2,32,224,224)).cuda()
        out = drop_space(batch_image)
    print("done")