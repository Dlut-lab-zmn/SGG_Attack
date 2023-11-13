# Copyright (c) Mn,Zhao. All Rights Reserved.
def grad(image, loss, model):
    image = image
    #计算梯度
    # Zero all existing gradients
    model.zero_grad()
    loss.backward()

    grad = image.grad
    # print(grad)
    return grad.reshape(image.shape)

def get_grad(image, loss, model):
    return grad(image, loss, model)
