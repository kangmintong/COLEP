import torch
import torch.optim as optim
import torch.nn.functional as F

def conformal_attack(x,y,model,max_norm,steps=20, smoothadv=False):
    batch_size = x.shape[0]
    multiplier = -1
    delta = torch.zeros_like(x, requires_grad=True).cuda()


    # Setup optimizers
    optimizer = optim.SGD([delta], lr=max_norm /steps * 2)

    for i in range(steps):
        if smoothadv:
            x = x + torch.randn_like(x).cuda() * 0.12
        adv = x + delta

        logits = model(adv)
        pred_labels = logits.argmax(1)
        ce_loss = F.cross_entropy(logits, y, reduction='sum')
        loss = multiplier * ce_loss

        optimizer.zero_grad()
        loss.backward()
        # renorming gradient
        # print(f'delta.grad.shape:{delta.grad.shape}')
        grad_norms = delta.grad.contiguous().view(batch_size, -1).norm(p=2, dim=1)
        delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

        optimizer.step()

        delta.data.renorm_(p=2, dim=0, maxnorm=max_norm)
    return x + delta

def conformal_attack_binary(x,y,model,max_norm,steps=20):
    batch_size = x.shape[0]
    multiplier = -1
    delta = torch.zeros_like(x, requires_grad=True).cuda()

    # Setup optimizers
    optimizer = optim.SGD([delta], lr=max_norm /steps * 2)

    for i in range(steps):
        adv = x + delta

        # logits = model(adv)

        logits = torch.zeros((x.shape[0],len(model))).cuda()
        for j in range(len(model)):
            logit = model[j](adv)[:,1].sigmoid()
            logits[:,j] = logit
        for ii in range(logits.shape[0]):
            sum_ = logits[ii].sum()
            logits[ii] /= sum_

        pred_labels = logits.argmax(1)

        # print(f'logits.shape:{logits.shape}')
        # print(f'y.shape:{y.shape}')
        # print(y)

        ce_loss = F.cross_entropy(logits, y, reduction='sum')
        loss = multiplier * ce_loss

        optimizer.zero_grad()
        loss.backward()
        # renorming gradient
        # print(f'delta.grad.shape:{delta.grad.shape}')
        grad_norms = delta.grad.contiguous().view(batch_size, -1).norm(p=2, dim=1)
        delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

        optimizer.step()

        delta.data.renorm_(p=2, dim=0, maxnorm=max_norm)
    return x + delta

def conformal_attack_knowledge(x,y,model,GT_attr,models_attr,max_norm=0.5,steps=20):
    batch_size = x.shape[0]
    multiplier = -1
    delta = torch.zeros_like(x, requires_grad=True).cuda()

    # Setup optimizers
    optimizer = optim.SGD([delta], lr=max_norm / steps * 2)

    for i in range(steps):

        adv = x + delta

        # logits = model(adv)

        logits = torch.zeros((x.shape[0], len(model))).cuda()
        for j in range(len(model)):
            logit = model[j](adv)[:, 1].sigmoid()
            logits[:, j] = logit
        for ii in range(logits.shape[0]):
            sum_ = logits[ii].sum()
            logits[ii] /= sum_

        pred_labels = logits.argmax(1)

        # print(f'logits.shape:{logits.shape}')
        # print(f'y.shape:{y.shape}')
        # print(y)

        ce_loss = F.cross_entropy(logits, y, reduction='sum')
        loss = multiplier * ce_loss

        logits_2 = torch.zeros((x.shape[0], len(models_attr))).cuda()
        for j in range(len(models_attr)):
            logit = models_attr[j](adv)[:, 1]
            logits_2[:, j] = logit

        bce_loss = F.binary_cross_entropy_with_logits(logits_2,GT_attr.float(),reduction='sum')
        loss += multiplier * bce_loss

        optimizer.zero_grad()
        loss.backward()
        # renorming gradient
        # print(f'delta.grad.shape:{delta.grad.shape}')
        grad_norms = delta.grad.contiguous().view(batch_size, -1).norm(p=2, dim=1)
        delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

        optimizer.step()

        delta.data.renorm_(p=2, dim=0, maxnorm=max_norm)
    return x + delta

def conformal_attack_pc(x,model_pc_reasoning,GT,GT_attr,max_norm=0.5,steps=20,num_class=12,smoothadv=False):
    batch_size = x.shape[0]
    multiplier = -1
    delta = torch.zeros_like(x, requires_grad=True).cuda()

    # Setup optimizers
    optimizer = optim.SGD([delta], lr=max_norm / steps * 2)


    for i in range(steps):
        if smoothadv:
            x = x + torch.randn_like(x).cuda() * 0.12
        adv = x + delta

        logits = model_pc_reasoning(adv)

        ce_loss = F.cross_entropy(logits[:,:num_class,1], GT, reduction='sum')
        loss = multiplier * ce_loss

        bce_loss = F.binary_cross_entropy_with_logits(logits[:,num_class:,1],GT_attr.float(),reduction='sum')
        loss += multiplier * bce_loss

        optimizer.zero_grad()
        loss.backward()
        # renorming gradient
        # print(f'delta.grad.shape:{delta.grad.shape}')
        grad_norms = delta.grad.contiguous().view(batch_size, -1).norm(p=2, dim=1)
        delta.grad.div_(grad_norms.view(-1, 1, 1, 1))

        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

        optimizer.step()

        delta.data.renorm_(p=2, dim=0, maxnorm=max_norm)
    return x + delta