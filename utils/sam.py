'''
@Description: 
@Author: shiyuan
@Date: 2021-03-13 14:56:57
@LastEditTime: 2021-03-13 16:57:40
@LastEditors: shiyuan
'''
import jittor as jt
from jittor import nn, Module
from jittor.optim import Optimizer
class SAM(Optimizer):
    '''
    SAM优化器
    '''
    def __init__(self, base_optimizer, lr,  rho = 0.05,  momentum=0, weight_decay=0, dampening=0, nesterov=False):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho)
        super(SAM, self).__init__(base_optimizer.param_groups, lr, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho

    def first_step(self, loss=None, zero_grad=True):
        current_lr=self.base_optimizer.param_groups[0].get("lr", self.base_optimizer.lr)


        if loss is not None:
            self.backward(loss)
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = (self.rho / (grad_norm + 1e-12)).minimum(current_lr*2)

            if "e_w" not in group:
                group["e_w"] = [ jt.zeros_like(p).stop_grad().stop_fuse() for p in group['params'] ]
            pg_e_w = group["e_w"]
            i = 0
            for p, g in zip(group["params"], group["grads"]):
                e_w = g * scale
                pg_e_w[i].update(e_w)
                i = i + 1
        for pg in self.param_groups:
            for p,  e_w in zip(pg["params"], pg["e_w"]):
                if p.is_stop_grad(): continue
                #p = p + e_w
                p.update(p + e_w)
                #p.update(p + e_w - g * lr)
        self.zero_grad()

    def second_step(self, loss=None, zero_grad=True):
        if loss is not None:
            self.backward(loss)
        for group in self.param_groups:
            for p,  e_w in zip(group["params"], group["e_w"]):
                if p.is_stop_grad(): continue
                p = p - e_w
                #p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        #if zero_grad: self.zero_grad()
            
    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")
    def _grad_norm(self):

        norm = jt.norm(
                    jt.contrib.concat([
                        jt.norm(jt.flatten(g), 2, 0)
                        for group in self.param_groups for p, g in zip(group["params"], group["grads"])
                    ]),
                    2 ,
                    0 ,
               )
        return norm

