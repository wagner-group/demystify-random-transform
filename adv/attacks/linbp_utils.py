'''
This code is used for LinBP attack and is adapted from
https://github.com/qizhangli/linbp-attack
'''

import torch
import torch.nn.functional as F


def linbp_forw_resnet(model, x, do_linbp, linbp_layer, arch):
    jj = int(linbp_layer.split('_')[0])
    kk = int(linbp_layer.split('_')[1])
    # NOTE: we don't use normalization
    # x = model[0](x)
    x = model.conv1(x)
    if arch == 'resnet50':
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
    ori_mask_ls = []
    conv_out_ls = []
    relu_out_ls = []
    conv_input_ls = []

    def layer_forw(jj, kk, jj_now, kk_now, x, mm, ori_mask_ls, conv_out_ls,
                   relu_out_ls, conv_input_ls, do_linbp):
        if jj < jj_now:
            x, ori_mask, conv_out, relu_out, conv_in = block_func3(
                mm, x, linbp=True)
            ori_mask_ls.append(ori_mask)
            conv_out_ls.append(conv_out)
            relu_out_ls.append(relu_out)
            conv_input_ls.append(conv_in)
        elif jj == jj_now:
            if kk_now >= kk:
                x, ori_mask, conv_out, relu_out, conv_in = block_func3(
                    mm, x, linbp=True)
                ori_mask_ls.append(ori_mask)
                conv_out_ls.append(conv_out)
                relu_out_ls.append(relu_out)
                conv_input_ls.append(conv_in)
            else:
                x, _, _, _, _ = block_func3(mm, x, linbp=False)
        else:
            x, _, _, _, _ = block_func3(mm, x, linbp=False)
        return x, ori_mask_ls

    for ind, mm in enumerate(model.layer1):
        x, ori_mask_ls = layer_forw(
            jj, kk, 1, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls,
            conv_input_ls, do_linbp)
    for ind, mm in enumerate(model.layer2):
        x, ori_mask_ls = layer_forw(
            jj, kk, 2, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls,
            conv_input_ls, do_linbp)
    for ind, mm in enumerate(model.layer3):
        x, ori_mask_ls = layer_forw(
            jj, kk, 3, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls,
            conv_input_ls, do_linbp)
    for ind, mm in enumerate(model.layer4):
        x, ori_mask_ls = layer_forw(
            jj, kk, 4, ind, x, mm, ori_mask_ls, conv_out_ls, relu_out_ls,
            conv_input_ls, do_linbp)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    x = model.fc(x)
    return x, (ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls)


# def block_func2(block, x, linbp):
#     identity = x
#     conv_in = x + 0
#     out = block.bn1(conv_in)
#     out_0 = out + 0
#     if linbp:
#         out = linbp_relu(out_0)
#     else:
#         out = F.relu(out_0)
#     ori_mask_0 = out.data.bool().int()
#     out = block.conv1(out)

#     out = block.bn2(out)
#     out_1 = out + 0
#     if linbp:
#         out = linbp_relu(out_1)
#     else:
#         out = F.relu(out_1)
#     ori_mask_1 = out.data.bool().int()
#     out = block.conv2(out)

#     if block.downsample is not None:
#         identity = block.downsample(identity)
#     identity_out = identity + 0
#     x_out = out + 0

#     out = identity_out + x_out
#     return out, (ori_mask_0, ori_mask_1), (identity_out, x_out), (out_0, out_1), (0, conv_in)


def block_func3(block, x, linbp):
    identity = x
    conv_in = x + 0
    out = block.conv1(conv_in)
    out = block.bn1(out)
    out_0 = out + 0
    if linbp:
        out = linbp_relu(out_0)
    else:
        out = block.relu(out_0)
    ori_mask_0 = out.data.bool().int()

    out = block.conv2(out)
    out = block.bn2(out)
    out_1 = out + 0
    if linbp:
        out = linbp_relu(out_1)
    else:
        out = block.relu(out_1)
    ori_mask_1 = out.data.bool().int()

    out = block.conv3(out)
    out = block.bn3(out)

    if block.downsample is not None:
        identity = block.downsample(identity)
    identity_out = identity + 0
    x_out = out + 0

    out = identity_out + x_out
    out = block.relu(out)
    ori_mask_2 = out.data.bool().int()
    return out, (ori_mask_0, ori_mask_1, ori_mask_2), (identity_out, x_out), (out_0, out_1), (0, conv_in)


def linbp_relu(x):
    x_p = F.relu(-x)
    x = x + x_p.data
    return x


def linbp_backw_resnet(img, loss, forw_outputs, xp):
    ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = forw_outputs
    for i in range(-1, - len(conv_out_ls) - 1, -1):
        if i == -1:
            grads = torch.autograd.grad(loss, conv_out_ls[i])
        else:
            grads = torch.autograd.grad(
                (conv_out_ls[i + 1][0], conv_input_ls[i + 1][1]),
                conv_out_ls[i], grad_outputs=(grads[0], main_grad_norm))
        normal_grad_2 = torch.autograd.grad(
            conv_out_ls[i][1], relu_out_ls[i][1], grads[1] * ori_mask_ls[i][2],
            retain_graph=True)[0]
        normal_grad_1 = torch.autograd.grad(
            relu_out_ls[i][1], relu_out_ls[i][0],
            normal_grad_2 * ori_mask_ls[i][1], retain_graph=True)[0]
        normal_grad_0 = torch.autograd.grad(
            relu_out_ls[i][0], conv_input_ls[i][1],
            normal_grad_1 * ori_mask_ls[i][0], retain_graph=True)[0]
        del normal_grad_2, normal_grad_1
        main_grad = torch.autograd.grad(
            conv_out_ls[i][1], conv_input_ls[i][1], grads[1])[0]
        alpha = normal_grad_0.norm(p=2, dim=(1, 2, 3), keepdim=True) \
            / main_grad.norm(p=2, dim=(1, 2, 3), keepdim=True)
        main_grad_norm = xp * alpha * main_grad
    input_grad = torch.autograd.grad(
        (conv_out_ls[0][0], conv_input_ls[0][1]), img,
        grad_outputs=(grads[0], main_grad_norm))
    return input_grad[0].data
