# import os
# import matplotlib.pyplot as plt
# import xml.etree.ElementTree as ET
# from xml.dom import minidom
# plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
# plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
#
# classes = [u"一次性快餐盒",u"书籍纸张",u"充电宝",u"剩饭剩菜",u"包",u"垃圾桶",u"塑料器皿",u"塑料玩具",u"塑料衣架",u"大骨头",u"干电池",u"快递纸袋",u"插头电线",u"旧衣服",
#           u"易拉罐",u"枕头",u"果皮果肉",u"毛绒玩具",u"污损塑料",u"污损用纸",u"洗护用品",u"烟蒂",u"牙签",u"玻璃器皿",u"砧板",u"筷子",u"纸盒纸箱",u"花盆",u"茶叶渣",u"菜帮菜叶",
#           u"蛋壳",u"调料瓶",u"软膏",u"过期药物",u"酒瓶",u"金属厨具",u"金属器皿",u"金属食品罐",u"锅",u"陶瓷器皿",u"鞋",u"食用油桶",u"饮料瓶",u"鱼骨"]
# num = [0 for i in range(44)]
# dir = os.listdir("trainval_garbage/garbage2020/Annotations/")
# for i in dir:
#     file = open("trainval_garbage/garbage2020/Annotations/" + i,'r',encoding='utf-8')
#     tree = ET.parse(file) #直接用这个读取.xml文件，不要用 open 打开再读，就不会出现中文乱码
#     root = tree.getroot()
#
#     for obj in root.iter('object'):
#         difficult = obj.find('difficult').text
#         cls = obj.find('name').text
#         # 检查该图片是否为voc里面的类别
#         if cls not in classes or int(difficult) == 1:
#             continue
#         cls_id = classes.index(cls)
#         num[cls_id] = num[cls_id] + 1
#
# # plt.hist(range(len(num)),num, bins=44, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
# plt.xticks(rotation=45)
# plt.bar(range(len(num)), num,color = 'rgb',label = classes,width=0.8,align="center")
# plt.show()

# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from torch.optim import lr_scheduler
#
# model = torch.nn.Sequential(
#     nn.Linear(10,100),
#     nn.Linear(100,10)
# )
#
# lr = 1e-3
# optimizer = torch.optim.Adam(model.parameters(),lr = lr)
# scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=7,T_mult=2,eta_min=0)
#
# x = list(range(200))
# y = []
# for epoch in range(200):
#     scheduler.step()
#     lr = scheduler.get_lr()
#     y.append(scheduler.get_lr()[0])
# plt.figure()
# plt.plot(x, y)
# plt.show()

# import math
# import warnings
# import torch
# import torch.nn as nn
# from torch.optim import Optimizer
# model = torch.nn.Sequential(
#     nn.Linear(10,100),
#     nn.Linear(100,10)
# )
#
# lr = 1e-3
# optimizer = torch.optim.SGD(model.parameters(),lr = lr)
# torch.optim.sgd
# class CosineAnnealingLR(Optimizer):
#     r"""Set the learning rate of each parameter group using a cosine annealing
#     schedule, where :math:`\eta_{max}` is set to the initial lr and
#     :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
#     .. math::
#         \begin{aligned}
#             \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
#             + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
#             & T_{cur} \neq (2k+1)T_{max}; \\
#             \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
#             \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
#             & T_{cur} = (2k+1)T_{max}.
#         \end{aligned}
#     When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
#     is defined recursively, the learning rate can be simultaneously modified
#     outside this scheduler by other operators. If the learning rate is set
#     solely by this scheduler, the learning rate at each step becomes:
#     .. math::
#         \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
#         \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
#     It has been proposed in
#     `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
#     implements the cosine annealing part of SGDR, and not the restarts.
#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         T_max (int): Maximum number of iterations.
#         eta_min (float): Minimum learning rate. Default: 0.
#         last_epoch (int): The index of last epoch. Default: -1.
#         verbose (bool): If ``True``, prints a message to stdout for
#             each update. Default: ``False``.
#     .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
#         https://arxiv.org/abs/1608.03983
#     """

    # def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
    #     self.T_max = T_max
    #     self.eta_min = eta_min
    #     super(CosineAnnealingLR, self).__init__(optimizer, last_epoch, verbose)
    #
    # def get_lr(self):
    #     if not self._get_lr_called_within_step:
    #         warnings.warn("To get the last learning rate computed by the scheduler, "
    #                       "please use `get_last_lr()`.", UserWarning)
    #
    #     if self.last_epoch == 0:
    #         return self.base_lrs
    #     elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
    #         return [group['lr'] + (base_lr - self.eta_min) *
    #                 (1 - math.cos(math.pi / self.T_max)) / 2
    #                 for base_lr, group in
    #                 zip(self.base_lrs, self.optimizer.param_groups)]
    #     return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
    #             (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
    #             (group['lr'] - self.eta_min) + self.eta_min
    #             for group in self.optimizer.param_groups]
    #
    # def _get_closed_form_lr(self):
    #     return [self.eta_min + (base_lr - self.eta_min) *
    #             (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
#                 for base_lr in self.base_lrs]
# t_0 = 1
# t_m = 1
# CosineAnnealingLR(optimizer,10,False)
# print(c)


# import types
# import math
# from torch._six import inf
# from functools import wraps
# import warnings
# import weakref
# from collections import Counter
# from bisect import bisect_right
#
# from torch.optim import Optimizer
#
# class _LRScheduler(object):
#
#     def __init__(self, optimizer, last_epoch=-1, verbose=False):
#
#         # Attach optimizer
#         if not isinstance(optimizer, Optimizer):
#             raise TypeError('{} is not an Optimizer'.format(
#                 type(optimizer).__name__))
#         self.optimizer = optimizer
#
#         # Initialize epoch and base learning rates
#         if last_epoch == -1:
#             for group in optimizer.param_groups:
#                 group.setdefault('initial_lr', group['lr'])
#         else:
#             for i, group in enumerate(optimizer.param_groups):
#                 if 'initial_lr' not in group:
#                     raise KeyError("param 'initial_lr' is not specified "
#                                    "in param_groups[{}] when resuming an optimizer".format(i))
#         self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
#         self.last_epoch = last_epoch
#
#         # Following https://github.com/pytorch/pytorch/issues/20124
#         # We would like to ensure that `lr_scheduler.step()` is called after
#         # `optimizer.step()`
#         def with_counter(method):
#             if getattr(method, '_with_counter', False):
#                 # `optimizer.step()` has already been replaced, return.
#                 return method
#
#             # Keep a weak reference to the optimizer instance to prevent
#             # cyclic references.
#             instance_ref = weakref.ref(method.__self__)
#             # Get the unbound method for the same purpose.
#             func = method.__func__
#             cls = instance_ref().__class__
#             del method
#
#             @wraps(func)
#             def wrapper(*args, **kwargs):
#                 instance = instance_ref()
#                 instance._step_count += 1
#                 wrapped = func.__get__(instance, cls)
#                 return wrapped(*args, **kwargs)
#
#             # Note that the returned function here is no longer a bound method,
#             # so attributes like `__func__` and `__self__` no longer exist.
#             wrapper._with_counter = True
#             return wrapper
#
#         self.optimizer.step = with_counter(self.optimizer.step)
#         self.optimizer._step_count = 0
#         self._step_count = 0
#         self.verbose = verbose
#
#         self.step()
#
#     def state_dict(self):
#         """Returns the state of the scheduler as a :class:`dict`.
#         It contains an entry for every variable in self.__dict__ which
#         is not the optimizer.
#         """
#         return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
#
#     def load_state_dict(self, state_dict):
#         """Loads the schedulers state.
#         Arguments:
#             state_dict (dict): scheduler state. Should be an object returned
#                 from a call to :meth:`state_dict`.
#         """
#         self.__dict__.update(state_dict)
#
#     def get_last_lr(self):
#         """ Return last computed learning rate by current scheduler.
#         """
#         return self._last_lr
#
#     def get_lr(self):
#         # Compute learning rate using chainable form of the scheduler
#         raise NotImplementedError
#
#     def print_lr(self, is_verbose, group, lr, epoch=None):
#         """Display the current learning rate.
#         """
#         if is_verbose:
#             if epoch is None:
#                 print('Adjusting learning rate'
#                       ' of group {} to {:.4e}.'.format(group, lr))
#             else:
#                 print('Epoch {:5d}: adjusting learning rate'
#                       ' of group {} to {:.4e}.'.format(epoch, group, lr))
#
#
#     def step(self, epoch=None):
#         # Raise a warning if old pattern is detected
#         # https://github.com/pytorch/pytorch/issues/20124
#         if self._step_count == 1:
#             if not hasattr(self.optimizer.step, "_with_counter"):
#                 warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
#                               "initialization. Please, make sure to call `optimizer.step()` before "
#                               "`lr_scheduler.step()`. See more details at "
#                               "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
#
#             # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
#             elif self.optimizer._step_count < 1:
#                 warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
#                               "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
#                               "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
#                               "will result in PyTorch skipping the first value of the learning rate schedule. "
#                               "See more details at "
#                               "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
#         self._step_count += 1
#
#         class _enable_get_lr_call:
#
#             def __init__(self, o):
#                 self.o = o
#
#             def __enter__(self):
#                 self.o._get_lr_called_within_step = True
#                 return self
#
#             def __exit__(self, type, value, traceback):
#                 self.o._get_lr_called_within_step = False
#
#         with _enable_get_lr_call(self):
#             if epoch is None:
#                 self.last_epoch += 1
#                 values = self.get_lr()
#             else:
#                 # warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
#                 self.last_epoch = epoch
#                 if hasattr(self, "_get_closed_form_lr"):
#                     values = self._get_closed_form_lr()
#                 else:
#                     values = self.get_lr()
#
#         for i, data in enumerate(zip(self.optimizer.param_groups, values)):
#             param_group, lr = data
#             param_group['lr'] = lr
#             self.print_lr(self.verbose, i, lr, epoch)
#
#         self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
# #
# #
# class CosineAnnealingWarmRestarts(_LRScheduler):
#
#     def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
#         if T_0 <= 0 or not isinstance(T_0, int):
#             raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
#         if T_mult < 1 or not isinstance(T_mult, int):
#             raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
#         self.T_0 = T_0
#         self.T_i = T_0
#         self.T_mult = T_mult
#         self.eta_min = eta_min
#
#         super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)
#
#         self.T_cur = self.last_epoch
#
#     def get_lr(self):
#         if not self._get_lr_called_within_step:
#             warnings.warn("To get the last learning rate computed by the scheduler, "
#                           "please use `get_last_lr()`.", UserWarning)
#
#         return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
#                 for base_lr in self.base_lrs]
#
#     def step(self, epoch=None):
#         """Step could be called after every batch update
#         Example:
#             >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
#             >>> iters = len(dataloader)
#             >>> for epoch in range(20):
#             >>>     for i, sample in enumerate(dataloader):
#             >>>         inputs, labels = sample['inputs'], sample['labels']
#             >>>         optimizer.zero_grad()
#             >>>         outputs = net(inputs)
#             >>>         loss = criterion(outputs, labels)
#             >>>         loss.backward()
#             >>>         optimizer.step()
#             >>>         scheduler.step(epoch + i / iters)
#         This function can be called in an interleaved way.
#         Example:
#             >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
#             >>> for epoch in range(20):
#             >>>     scheduler.step()
#             >>> scheduler.step(26)
#             >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
#         """
#
#         if epoch is None and self.last_epoch < 0:
#             epoch = 0
#
#         if epoch is None:
#             epoch = self.last_epoch + 1
#             self.T_cur = self.T_cur + 1
#             if self.T_cur >= self.T_i:
#                 self.T_cur = self.T_cur - self.T_i
#                 self.T_i = self.T_i * self.T_mult
#         else:
#             if epoch < 0:
#                 raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
#             if epoch >= self.T_0:
#                 if self.T_mult == 1:
#                     self.T_cur = epoch % self.T_0
#                 else:
#                     n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
#                     self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
#                     self.T_i = self.T_0 * self.T_mult ** (n)
#             else:
#                 self.T_i = self.T_0
#                 self.T_cur = epoch
#         self.last_epoch = math.floor(epoch)
#
#         class _enable_get_lr_call:
#
#             def __init__(self, o):
#                 self.o = o
#
#             def __enter__(self):
#                 self.o._get_lr_called_within_step = True
#                 return self
#
#             def __exit__(self, type, value, traceback):
#                 self.o._get_lr_called_within_step = False
#                 return self
#
#         with _enable_get_lr_call(self):
#             for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
#                 param_group, lr = data
#                 param_group['lr'] = lr
#                 self.print_lr(self.verbose, i, lr, epoch)
#
#         self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
#
# optimizer = torch.optim.SGD(model.parameters(),lr = 1e-3,momentum=0.9)
# lr_sec = CosineAnnealingWarmRestarts(optimizer,T_0=7,T_mult=1)
# print(lr_sec)

# import torch
# from torch.optim import Optimizer  #, required
#
# class _RequiredParameter(object):
#     """Singleton class representing a required parameter for an Optimizer."""
#     def __repr__(self):
#         return "<required parameter>"
#
# required = _RequiredParameter()
#
# class SGD(Optimizer):
#
#     def __init__(self, params, lr=required, momentum=0, dampening=0,
#                  weight_decay=0, nesterov=False):
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if momentum < 0.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#
#         defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
#                         weight_decay=weight_decay, nesterov=nesterov)
#         if nesterov and (momentum <= 0 or dampening != 0):
#             raise ValueError("Nesterov momentum requires a momentum and zero dampening")
#         super(SGD, self).__init__(params, defaults)
#
#     def __setstate__(self, state):
#         super(SGD, self).__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('nesterov', False)
#
#     @torch.no_grad()
#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()
#
#         for group in self.param_groups:
#             weight_decay = group['weight_decay']
#             momentum = group['momentum']
#             dampening = group['dampening']
#             nesterov = group['nesterov']
#
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 d_p = p.grad
#                 if weight_decay != 0:
#                     d_p = d_p.add(p, alpha=weight_decay)
#                 if momentum != 0:
#                     param_state = self.state[p]
#                     if 'momentum_buffer' not in param_state:
#                         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
#                     else:
#                         buf = param_state['momentum_buffer']
#                         buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
#                     if nesterov:
#                         d_p = d_p.add(buf, alpha=momentum)
#                     else:
#                         d_p = buf
#
#                 p.add_(d_p, alpha=-group['lr'])
#
#         return loss
#
# SGD(model.parameters(), lr=0.1, momentum=0.9)
#
#
#
# class _LRScheduler(object):
#
#     def __init__(self, optimizer, last_epoch=-1, verbose=False):
#
#         # Attach optimizer
#         if not isinstance(optimizer, Optimizer):
#             raise TypeError('{} is not an Optimizer'.format(
#                 type(optimizer).__name__))
#         self.optimizer = optimizer
#
#         # Initialize epoch and base learning rates
#         if last_epoch == -1:
#             for group in optimizer.param_groups:
#                 group.setdefault('initial_lr', group['lr'])
#         else:
#             for i, group in enumerate(optimizer.param_groups):
#                 if 'initial_lr' not in group:
#                     raise KeyError("param 'initial_lr' is not specified "
#                                    "in param_groups[{}] when resuming an optimizer".format(i))
#         self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
#         self.last_epoch = last_epoch
#
#         # Following https://github.com/pytorch/pytorch/issues/20124
#         # We would like to ensure that `lr_scheduler.step()` is called after
#         # `optimizer.step()`
#         def with_counter(method):
#             if getattr(method, '_with_counter', False):
#                 # `optimizer.step()` has already been replaced, return.
#                 return method
#
#             # Keep a weak reference to the optimizer instance to prevent
#             # cyclic references.
#             instance_ref = weakref.ref(method.__self__)
#             # Get the unbound method for the same purpose.
#             func = method.__func__
#             cls = instance_ref().__class__
#             del method
#
#             @wraps(func)
#             def wrapper(*args, **kwargs):
#                 instance = instance_ref()
#                 instance._step_count += 1
#                 wrapped = func.__get__(instance, cls)
#                 return wrapped(*args, **kwargs)
#
#             # Note that the returned function here is no longer a bound method,
#             # so attributes like `__func__` and `__self__` no longer exist.
#             wrapper._with_counter = True
#             return wrapper
#
#         self.optimizer.step = with_counter(self.optimizer.step)
#         self.optimizer._step_count = 0
#         self._step_count = 0
#         self.verbose = verbose
#
#         self.step()
#
#     def state_dict(self):
#         """Returns the state of the scheduler as a :class:`dict`.
#         It contains an entry for every variable in self.__dict__ which
#         is not the optimizer.
#         """
#         return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
#
#     def load_state_dict(self, state_dict):
#         """Loads the schedulers state.
#         Arguments:
#             state_dict (dict): scheduler state. Should be an object returned
#                 from a call to :meth:`state_dict`.
#         """
#         self.__dict__.update(state_dict)
#
#     def get_last_lr(self):
#         """ Return last computed learning rate by current scheduler.
#         """
#         return self._last_lr
#
#     def get_lr(self):
#         # Compute learning rate using chainable form of the scheduler
#         raise NotImplementedError
#
#     def print_lr(self, is_verbose, group, lr, epoch=None):
#         """Display the current learning rate.
#         """
#         if is_verbose:
#             if epoch is None:
#                 print('Adjusting learning rate'
#                       ' of group {} to {:.4e}.'.format(group, lr))
#             else:
#                 print('Epoch {:5d}: adjusting learning rate'
#                       ' of group {} to {:.4e}.'.format(epoch, group, lr))
#
#
#     def step(self, epoch=None):
#         # Raise a warning if old pattern is detected
#         # https://github.com/pytorch/pytorch/issues/20124
#         if self._step_count == 1:
#             if not hasattr(self.optimizer.step, "_with_counter"):
#                 warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
#                               "initialization. Please, make sure to call `optimizer.step()` before "
#                               "`lr_scheduler.step()`. See more details at "
#                               "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
#
#             # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
#             elif self.optimizer._step_count < 1:
#                 warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
#                               "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
#                               "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
#                               "will result in PyTorch skipping the first value of the learning rate schedule. "
#                               "See more details at "
#                               "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
#         self._step_count += 1
#
#         class _enable_get_lr_call:
#
#             def __init__(self, o):
#                 self.o = o
#
#             def __enter__(self):
#                 self.o._get_lr_called_within_step = True
#                 return self
#
#             def __exit__(self, type, value, traceback):
#                 self.o._get_lr_called_within_step = False
#
#         with _enable_get_lr_call(self):
#             if epoch is None:
#                 self.last_epoch += 1
#                 values = self.get_lr()
#             else:
#                 # warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
#                 self.last_epoch = epoch
#                 if hasattr(self, "_get_closed_form_lr"):
#                     values = self._get_closed_form_lr()
#                 else:
#                     values = self.get_lr()
#
#         for i, data in enumerate(zip(self.optimizer.param_groups, values)):
#             param_group, lr = data
#             param_group['lr'] = lr
#             self.print_lr(self.verbose, i, lr, epoch)
#
#         self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

# import torch
# from torch.optim import Optimizer
#
# class ExponentialLR(_LRScheduler):
#     """Decays the learning rate of each parameter group by gamma every epoch.
#     When last_epoch=-1, sets initial lr as lr.
#
#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         gamma (float): Multiplicative factor of learning rate decay.
#         last_epoch (int): The index of last epoch. Default: -1.
#     """

#     def __init__(self, optimizer, gamma, last_epoch=-1):
#         self.gamma = gamma
#         super(ExponentialLR, self).__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         if not self._get_lr_called_within_step:
#             warnings.warn("To get the last learning rate computed by the scheduler, "
#                           "please use `get_last_lr()`.", UserWarning)
#
#         if self.last_epoch == 0:
#             return self.base_lrs
#         return [group['lr'] * self.gamma
#                 for group in self.optimizer.param_groups]
#
#     def _get_closed_form_lr(self):
#         return [base_lr * self.gamma ** self.last_epoch
#                 for base_lr in self.base_lrs]
#
# optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
# ExponentialLR(optimizer,gamma=0.95)

# import cv2
# import os
# #
# pth = r"C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/trainval_garbage/garbage2020/JPEGImages/"
# dir = os.listdir(pth)
# for k,v in enumerate(dir):
#     img = cv2.imread(os.path.join(pth,v))
#     cv2.imshow("img",img)
#     cv2.waitKey(2500)
#     cv2.destroyAllWindows()
    # cv2.destroyWindow()

# torch.optim.Adam(model.parameters(),lr = 1e-3,N)


# img = cv2.imread(pth + "2b09f04c1b078b57980c0ac9cc18c6b.jpg")
# cv2.imshow('img',img)
# cv2.waitKey()


# import tensorflow as tf
# s = tf.Variable([[[1,3,2],[4,5,6],[7,8,9]],[[1,3,2],[4,5,6],[7,8,9]]], dtype=tf.float32)
# mean, variance = tf.nn.moments(s, [0,1])
# # init = tf.global_variables_initializer()
# print(s.shape)
# # print(init)
# print(mean)
# print(variance)
#([1,3,2] + [1,3,2]) / 3 = 4

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils import data
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# # from tensorboardX import SummaryWriter
#
# # define pytorch device - useful for device-agnostic execution
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # define model parameters
# NUM_EPOCHS = 90  # original paper
# BATCH_SIZE = 128
# MOMENTUM = 0.9
# LR_DECAY = 0.0005
# LR_INIT = 0.01
# IMAGE_DIM = 227  # pixels
# NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
# DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use
# # modify this to point to your data directory
# INPUT_ROOT_DIR = 'alexnet_data_in'
# TRAIN_IMG_DIR = 'alexnet_data_in/imagenet'
# OUTPUT_DIR = 'alexnet_data_out'
# LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
# CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints
#
# # make checkpoint path directory
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
#
# class AlexNet(nn.Module):
#     """
#     Neural network model consisting of layers propsed by AlexNet paper.
#     """
#     def __init__(self, num_classes=1000):
#         """
#         Define and allocate layers for this neural net.
#         Args:
#             num_classes (int): number of classes to predict with this model
#         """
#         super().__init__()
#         # input size should be : (b x 3 x 227 x 227)
#         # The image in the original paper states that width and height are 224 pixels, but
#         # the dimensions after first convolution layer do not lead to 55 x 55.
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
#             nn.ReLU(),
#             nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
#             nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
#             nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
#             nn.ReLU(),
#             nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
#             nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
#             nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
#             nn.ReLU(),
#             nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
#             nn.ReLU(),
#             nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
#         )
#         # classifier is just a name for linear layers
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5, inplace=True),
#             nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
#             nn.ReLU(),
#             nn.Dropout(p=0.5, inplace=True),
#             nn.Linear(in_features=4096, out_features=4096),
#             nn.ReLU(),
#             nn.Linear(in_features=4096, out_features=num_classes),
#         )
#         self.init_bias()  # initialize bias
#
#     def init_bias(self):
#         for layer in self.net:
#             if isinstance(layer, nn.Conv2d):
#                 nn.init.normal_(layer.weight, mean=0, std=0.01)
#                 nn.init.constant_(layer.bias, 0)
#         # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
#         nn.init.constant_(self.net[4].bias, 1)
#         nn.init.constant_(self.net[10].bias, 1)
#         nn.init.constant_(self.net[12].bias, 1)
#
#     def forward(self, x):
#         """
#         Pass the input through the net.
#         Args:
#             x (Tensor): input tensor
#         Returns:
#             output (Tensor): output tensor
#         """
#         x = self.net(x)
#         x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
#         return self.classifier(x)
#
#
# if __name__ == '__main__':
#     # print the seed value
#     seed = torch.initial_seed()
#     print('Used seed : {}'.format(seed))
#
#     tbwriter = SummaryWriter(log_dir=LOG_DIR)
#     print('TensorboardX summary writer created')
#
#     # create model
#     alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
#     # train on multiple GPUs
#     alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
#     print(alexnet)
#     print('AlexNet created')
#
#     # create dataset and data loader
#     dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
#         # transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
#         transforms.CenterCrop(IMAGE_DIM),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]))
#     print('Dataset created')
#     dataloader = data.DataLoader(
#         dataset,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=8,
#         drop_last=True,
#         batch_size=BATCH_SIZE)
#     print('Dataloader created')
#
#     # create optimizer
#     # the one that WORKS
#     optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
#     ### BELOW is the setting proposed by the original paper - which doesn't train....
#     # optimizer = optim.SGD(
#     #     params=alexnet.parameters(),
#     #     lr=LR_INIT,
#     #     momentum=MOMENTUM,
#     #     weight_decay=LR_DECAY)
#     print('Optimizer created')
#
#     # multiply LR by 1 / 10 after every 30 epochs
#     lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
#     print('LR Scheduler created')
#
#     # start training!!
#     print('Starting training...')
#     total_steps = 1
#     for epoch in range(NUM_EPOCHS):
#         lr_scheduler.step()
#         for imgs, classes in dataloader:
#             imgs, classes = imgs.to(device), classes.to(device)
#
#             # calculate the loss
#             output = alexnet(imgs)
#             loss = F.cross_entropy(output, classes)
#
#             # update the parameters
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # log the information and add to tensorboard
#             if total_steps % 10 == 0:
#                 with torch.no_grad():
#                     _, preds = torch.max(output, 1)
#                     accuracy = torch.sum(preds == classes)
#
#                     print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
#                         .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
#                     tbwriter.add_scalar('loss', loss.item(), total_steps)
#                     tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)
#
#             # print out gradient values and parameter average values
#             if total_steps % 100 == 0:
#                 with torch.no_grad():
#                     # print and save the grad of the parameters
#                     # also print and save parameter values
#                     print('*' * 10)
#                     for name, parameter in alexnet.named_parameters():
#                         if parameter.grad is not None:
#                             avg_grad = torch.mean(parameter.grad)
#                             print('\t{} - grad_avg: {}'.format(name, avg_grad))
#                             tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
#                             tbwriter.add_histogram('grad/{}'.format(name),
#                                     parameter.grad.cpu().numpy(), total_steps)
#                         if parameter.data is not None:
#                             avg_weight = torch.mean(parameter.data)
#                             print('\t{} - param_avg: {}'.format(name, avg_weight))
#                             tbwriter.add_histogram('weight/{}'.format(name),
#                                     parameter.data.cpu().numpy(), total_steps)
#                             tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)
#
#             total_steps += 1
#
#         # save checkpoints
#         checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
#         state = {
#             'epoch': epoch,
#             'total_steps': total_steps,
#             'optimizer': optimizer.state_dict(),
#             'model': alexnet.state_dict(),
#             'seed': seed,
#         }
#         torch.save(state, checkpoint_path)

# import torch.nn as nn
# nn.GroupNorm(3,64)

# from PIL import Image
#
# for i in range(10):
#     try:
#         img = Image.open(r"C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/trainval_garbage/garbage2020/ImageSets/img_1.jpg")
#         # image = Image.open(line[0])
#     except:
#         continue
#         # print("error !")
#     print("here")

# import torch.nn as nn
# import torch
# import numpy as np
# model = nn.Sequential(
#     nn.Linear(1,10),
#     nn.Linear(10,1)
# )
# optimizer = torch.optim.SGD(model.parameters(),lr = 1e-3,momentum=0.95)
# lr_sec = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3)
# #
# for k,v in enumerate(model.parameters()):
#     print("k:{},v:{}".format(k,v))
#     print("k.shape:{},v.shape:{}".format(np.shape(k),np.shape(v)))
#
# for k,v in enumerate(optimizer.state_dict()):
#     print("k:{},v:{}".format(k,v))
# #
# # print(optimizer.param_groups())
#
# # torch.nn.init.xavier_normal_(tensor, gain=1.0)

# from PIL import Image
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# p1 = r"C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/garbage_valid.txt"
# p3 = r"C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/garbage_img_valid.txt"
# p2 = r"F:/garbage_valid/"
# with open(p1,'r') as f1:
#     lines = f1.readlines()
#     f1.close()
#
# with open(p3,'r') as f2:
#     img_name = f2.readlines()
#     f2.close()

# for k,v in enumerate(lines):
#     # img = Image.open(v.strip())
#     # Image.save(p1,'{}'.format(img_name[k].strip()))
#     # pth = v.strip()
#     # img = cv2.imdecode(np.fromfile(v.strip()),1).astype('uint8')
#     # img = np.ascontiguousarray(img)
#     img = plt.imread(v.strip())
#     name = p2 + img_name[k].strip() ###再保存的名字前面加上路径！！！！
#     plt.imsave(name,img)
#     print("{}".format(img_name[k].strip()))
    # cv2.imwrite(name,p1,img)
    # img = plt.imread(v.strip())
    # plt.savefig(r"F:/garbage_valid/")
# img = []
# img_label = []
# img_save_path = r'F://valid_img.npy'
# label_save_path = r"F://valid_label.npy"
# for k,v in enumerate(lines):
#     image = plt.imread(v.strip())
#     image = image/255.
#     img.append(np.asarray(image))
#     img_label.append(k)
#     print(img_name[k].strip())
# img_save = np.reshape(img,(len(img),-1))
# label_save = np.reshape(img_label,(len(img_label),-1))
# np.save(img_save_path,img_save)
# np.save(label_save_path,label_save)
#
# img = np.load(img_save_path)
# label = np.load(label_save_path)
# import os
# txt_dir = os.listdir(r"C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/input/ground-truth/")
# print("txt_dir:{}".format(len(txt_dir)))
#
# with open(r"C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/input/no2/no2.txt",'r') as f:
#     no2_txt = f.readlines()
#     f.close()
# print(len(no2_txt))
# import os
# res_txt = os.listdir(r"C:/Users/WANGYONG/Desktop/pycharm/yolo3_pytorch_bb/input/detection-results/")
# print(len(res_txt))

#为图片添加高斯噪声并保存
# def add_gaussian_and_save(image):
#     gaussian_noise_imgs = [] #添加完高斯噪声之后的图片集
#     index = 0 #图片索引
#     # for image in images:
#     gaussian_noise_img = gaussian(image, 20)
#     index += 1
#     # cv2.imwrite('Image_Gaussian_Noise\\' + str(index) + '.jpg', gaussian_noise_img)
#     # gaussian_noise_imgs.append(gaussian_noise_img)
#     return gaussian_noise_img #返回添加高斯噪声之后的图片集
#
# #定义添加高斯噪声的函数,src灰度图片,scale噪声标准差
# def gaussian(src, scale):
#     gaussian_noise_img = np.copy(src) #深拷贝
#     noise = np.random.normal(0, scale, size=(src.size[1], src.size[0],3)) #噪声
#     add_noise_and_check = np.array(gaussian_noise_img, dtype=np.float32) #未经检查的图片
#     add_noise_and_check += noise
#     add_noise_and_check = add_noise_and_check
    # #原来的错误算法
    # # gaussian_noise_num = int(per * src.shape[0] * src.shape[1])
    # # for i in range(gaussian_noise_num):
    # #     rand_x = np.random.randint(0, src.shape[0])
    # #     rand_y = np.random.randint(0, src.shape[1])
    # #     #添加高斯噪声
    # #     gaussian_noise_img[rand_x, rand_y] += int(10 * np.random.randn()) #要添加的噪声数值
    # for i in range(len(add_noise_and_check)):
    #     for j in range(len(add_noise_and_check[0])):
    #         if add_noise_and_check[i][j].all() > 255:
    #             add_noise_and_check[i][j] = 255
    #         elif add_noise_and_check[i][j].all() < 0:
    #             add_noise_and_check[i][j] = 0
    # '''
    # uint8是无符号整数,0到255之间
    # 0黑,255白
    # 256等价于0,-1等价于255
    # 每256个数字一循环
    # '''
#     gaussian_noise_img = np.array(add_noise_and_check, dtype=np.uint8)
#     return gaussian_noise_img #返回添加了高斯噪声之后的图片
#
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# noise1 = np.random.normal(0, 1, size=(416, 416,3))
# print(noise1.shape)
# print(noise1)
# noise2 = np.random.randn(0,20,size = (416,416))
# print(noise2.shape)


# jitter = .3
# w ,h = (416,416)
# def Rand(a=0, b=1):
#     return np.random.rand() * (b - a) + a
# # 调整图片大小
# new_ar = w / h * Rand(1 - jitter, 1 + jitter) / Rand(1 - jitter, 1 + jitter)
# scale = Rand(.25, 2)
# if new_ar < 1:
#     nh = int(scale * h)
#     nw = int(nh * new_ar)
# else:
#     nw = int(scale * w)
#     nh = int(nw / new_ar)
# image = image.resize((nw, nh), Image.BICUBIC)
#
# # image = Image.fromarray(image)
# # image = image + noise
# # noise = np.random.randn(image.size(0),image.size(1))
# # 放置图片
# dx = int(Rand(0, w - nw))
# dy = int(Rand(0, h - nh))
# new_image = Image.new('RGB', (w, h),
#                       (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
# new_image.paste(image, (dx, dy))
# image = new_image
# image.show()
# # 添加高斯噪声
# noise = np.random.normal(0, 1, size=(image.size[1], image.size[0], 3))
# image = image + noise
#

# image = add_gaussian_and_save(image)
# print(image)
# image = np.array(image,dtype=np.uint8)
# image = Image.fromarray(image)
# image.show()
# image = Image.open(r"F:/garbage_valid/2b09f04c1b078b57980c0ac9cc18c6b.jpg")
# L = []
# h,w,c = np.shape(image)
# import numpy as np
# for i in range(3):
#     for a in range(w):
#         for b in range(h):
#             if image[a][b][i] == 255:
#                 L.append((a,b,i))
#
#         # print("\n")
# print(L)
# transforms.ToPILImage()(image).convert('RGB')
# 是否翻转图片
# flip = Rand() < .5
# if flip:
#     image = image.transpose(Image.FLIP_LEFT_RIGHT)

# import torch.nn as nn
# import torch
# m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# input = torch.randn(20, 16, 50, 100)
# output = m(input)
# print(output.shape)
# import os
# dir1 = os.listdir(r"C:/Users/WANGYONG/Desktop/互联网+/data/绿萝-百度/晒伤/")
#
# for k,v in enumerate(dir1):
#     src = "C:/Users/WANGYONG/Desktop/互联网+/data/绿萝-百度/晒伤/" + str(v)
#     dct = "C:/Users/WANGYONG/Desktop/互联网+/data/绿萝-百度/晒伤/" + "s_" + "5" + "_" + str(k+1)  + ".jpg"
#     os.rename(src,dct)

# d1 = "C:/Users/WANGYONG/Desktop/互联网+/data/绿萝-百度/tmp/"

# from skimage import util
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
#
# def gaussNoise(img):
#     noise_img = util.random_noise(img, mode='gaussian')
#     result = util.img_as_ubyte(noise_img)
#     result = np.array(result,dtype=np.uint8)
#     result = Image.fromarray(result)
#     return result
#
# img = Image.open(r"F:/garbage_valid/2b09f04c1b078b57980c0ac9cc18c6b.jpg")
# img = gaussNoise(np.array(img))
# img.show()
# plt.figure()
# plt.imshow("img",img)
# import os
# d1 = os.listdir(r"F:/input/new_12/detection-results/")
# print(len(d1))
# from PIL import Image
# import numpy as np
# def horizontalFlip(img):
#     size = img.shape  # 获得图像的形状
#     iLR = img.copy()  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
#     h = size[0]
#     w = size[1]
#     for i in range(h):  # 元素循环
#         for j in range(w):
#             iLR[i, w - 1 - j] = img[i, j]
#     iLR = np.array(iLR,dtype=np.uint8)
#     return iLR
#
#
# # 垂直翻转
# def verticalFlip(img):
#     size = img.shape  # 获得图像的形状
#     iLR = img.copy()  # 获得一个和原始图像相同的图像，注意这里要使用深度复制
#     h = size[0]
#     w = size[1]
#     for i in range(h):  # 元素循环
#         for j in range(w):
#             iLR[h - 1 - i, j] = img[i, j]
#     iLR = np.array(iLR, dtype=np.uint8)
#     return iLR
#
# img = Image.open(r"D:/software/新建文件夹 (2)/photo/蓝二/兰二.jpg")
#
# new1_img = horizontalFlip(np.array(img))
# new1_img = Image.fromarray(new1_img)
# new1_img.show()
#
# new2_img = verticalFlip(np.array(img))
# new2_img = Image.fromarray(new2_img)
# new2_img.show()
# import codecs
# def unpickle(file):
#     import pickle
#     with codecs.open(file, 'r') as fo:
#         dict = pickle.load(fo)
#     return dict

# file = r"F:/迅雷下载/cifar-10/data/cifar-10-batches-py/data_batch_1.npy"
# import numpy as np
# dict = np.load(file,allow_pickle=True)
# print(dict)

# import pickle as p
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
# def load_CIFAR_batch(filename):
#     """ 载入cifar数据集的一个batch """
#     with open(filename, 'rb') as f:
#         # data = f.read()
#         datadict = p.load(f,encoding="latin1")
#         X = datadict['data']
#         # print(X.shape)
#         Y = datadict['labels']
#         # print(Y.shape)
#         X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
#         Y = np.array(Y)
#         return X, Y
#
# def load_CIFAR10(ROOT):
#     """ 载入cifar全部数据 """
#     xs = []
#     ys = []
#     for b in range(1,6):
#         f = os.path.join(ROOT, 'data_batch_%d' % (b,))
#         X, Y = load_CIFAR_batch(f)
#         xs.append(X)
#         ys.append(Y)
#     Xtr = np.concatenate(xs)
#     Ytr = np.concatenate(ys)
#     del X, Y
#     Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
#     return Xtr, Ytr, Xte, Yte
#
# file = r"F:/迅雷下载/cifar-10/data/cifar-10-batches-py/"
#
# x_train,y_train,x_test,y_test = load_CIFAR10(file)
# # L = [[x_train],[y_train],[x_test],[y_test]]
# plt.figure()
# # for data_set in L:
# for i in range(10000):
#     img = x_test[i:i+1,:]
#     img = img[0]/255
#     # img = img.reshape(64,64,3)
#     # img = img.view(32,32,3)
#     plt.imsave("F:/迅雷下载/cifar-10/data/cifar-10-batches-py/test_image_cifar10/" + str(i+1) + ".jpg",img)
#     with open("F:/迅雷下载/cifar-10/data/cifar-10-batches-py/test_image_cifar10/" + str(i+1) +'.txt','w') as f:
#         f.write(str(y_test[i]) + " " + str(i+1) + ".jpg")
#         f.close()
#
# import os
# d1 = os.listdir("F:/迅雷下载/cifar-10/data/cifar-10-batches-py/test_image_cifar10/")
# print(len(d1))

# import random
# import matplotlib.pyplot as plt
# import os
# y=list(range(1,5000))
# slice = random.sample(y, 500)  #从list中随机获取5个元素，作为一个片断返回
# source_pth = r"F:/迅雷下载/cifar-10/data/cifar-10-batches-py/train_image_cifar10/"
# pth = r"F:/迅雷下载/cifar-10/data/cifar-10-batches-py/random_sample/"
#
# d1 = os.listdir(source_pth)

# for i in d1:
#     label = i.split("_")[0]
#     name = i.split("_")[1]
#     with open(source_pth + name.split(".")[0] + '.txt','w') as f:
#         f.write(str(label) + " " + name)
#         f.close()
#     os.rename(source_pth+i,source_pth+name)


'''for i in slice:
    img = plt.imread(source_pth + str(i) + ".jpg")
    line = []
    with open(source_pth + str(i) + ".txt",'r') as f:
        line = f.readline()
        f.close()

    with open(pth + str(i) + ".txt",'w') as f2:
        f2.write(line)
        f2.close()
    plt.imsave(pth + str(i) + ".jpg",img)

print (slice)
print (y)
'''
# import os
#
# d1 = os.listdir("F:/迅雷下载/cifar-10/data/cifar-10-batches-py/random_sample/")
# print(len(d1))
# str = '''
# [123]#0
# L = "a"
# '''
# L = []
# S = []
# for s in str:
#     if s == '[':
#         L.append([])
#     else:
#         S = s.split("=")
#         # key,value = s.split("=")
# print(s)
# print(key,value)

from PCV.tools import imtools
import pickle
from scipy import *
from pylab import *
from PIL import Image
from scipy.cluster.vq import *
from PCV.tools import pca

# Uses sparse pca codepath.
# imlist = imtools.get_imlist(r'F:/迅雷下载/cifar-10/data/cifar-10-batches-py/one_class/')
imlist = imtools.get_imlist(r'F:/迅雷下载/data/a_selected_thumbs/')
# 获取图像列表和他们的尺寸
im = array(Image.open(imlist[0]))  # open one image to get the size
m, n = im.shape[:2]  # get the size of the images
imnbr = len(imlist)  # get the number of images
print ("The number of images is %d" % imnbr)

# Create matrix to store all flattened images
immatrix = array([array(Image.open(imname)).flatten() for imname in imlist], 'f')

# PCA降维
V, S, immean = pca.pca(immatrix)

# 保存均值和主成分
#f = open('./a_pca_modes.pkl', 'wb')
f = open(r'F:/迅雷下载/cifar-10/data/cifar-10-batches-py/a_pca_modes.pkl', 'wb')
pickle.dump(immean,f)
pickle.dump(V,f)
f.close()


# get list of images
# imlist = imtools.get_imlist(r'F:/迅雷下载/cifar-10/data/cifar-10-batches-py/one_class/')
imlist = imtools.get_imlist(r'F:/迅雷下载/data/a_selected_thumbs/')
imnbr = len(imlist)

# load model file
with open(r'F:/迅雷下载/cifar-10/data/cifar-10-batches-py/a_pca_modes.pkl','rb') as f:
    immean = pickle.load(f)
    V = pickle.load(f)
# create matrix to store all flattened images
immatrix = array([array(Image.open(im)).flatten() for im in imlist],'f')

# project on the 40 first PCs
immean = immean.flatten()
projected = array([dot(V[:40],immatrix[i]-immean) for i in range(imnbr)])

# k-means
projected = whiten(projected)
centroids,distortion = kmeans(projected,3)
code,distance = vq(projected,centroids)

# plot clusters
for k in range(3):
    ind = where(code==k)[0]
    figure()
    gray()
    for i in range(minimum(len(ind),30)):
        subplot(3,10,i+1)
        imshow(immatrix[ind[i]].reshape((25,25)))
        axis('off')
show()

 # -*- coding: utf-8 -*-
# from PCV.tools import imtools, pca
# from PIL import Image, ImageDraw
# from pylab import *
# from PCV.clustering import  hcluster
#
# imlist = imtools.get_imlist(r'F:/迅雷下载/data/a_selected_thumbs/')
# imnbr = len(imlist)
#
# # Load images, run PCA.
# immatrix = array([array(Image.open(im)).flatten() for im in imlist], 'f')
# V, S, immean = pca.pca(immatrix)
#
# # Project on 2 PCs.
# projected = array([dot(V[[0, 1]], immatrix[i] - immean) for i in range(imnbr)])  # P131 Fig6-3左图
# #projected = array([dot(V[[1, 2]], immatrix[i] - immean) for i in range(imnbr)])  # P131 Fig6-3右图
#
# # height and width
# h, w = 1200, 1200
#
# # create a new image with a white background
# img = Image.new('RGB', (w, h), (255, 255, 255))
# draw = ImageDraw.Draw(img)
#
# # draw axis
# draw.line((0, h/2, w, h/2), fill=(255, 0, 0))
# draw.line((w/2, 0, w/2, h), fill=(255, 0, 0))
#
# # scale coordinates to fit
# scale = abs(projected).max(0)
# scaled = floor(array([(p/scale) * (w/2 - 20, h/2 - 20) + (w/2, h/2)
#                       for p in projected])).astype(int)
#
# # paste thumbnail of each image
# for i in range(imnbr):
#     nodeim = Image.open(imlist[i])
#     nodeim.thumbnail((25, 25))
#     ns = nodeim.size
#     box = (scaled[i][0] - ns[0] // 2, scaled[i][1] - ns[1] // 2,
#          scaled[i][0] + ns[0] // 2 + 1, scaled[i][1] + ns[1] // 2 + 1)
#     img.paste(nodeim, box)
#
# tree = hcluster.hcluster(projected)
# hcluster.draw_dendrogram(tree,imlist,filename='fonts.png')
#
# figure()
# imshow(img)
# axis('off')
# img.save('F:/迅雷下载/data/pca_font.png')
# show()













