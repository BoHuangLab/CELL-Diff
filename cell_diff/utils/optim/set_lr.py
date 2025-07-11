import math
from torch.optim import Adam, Optimizer
from cell_diff.logging.loggers import logger

def group_param(net, ndim, d_tilde, lr):
    if ndim < 1:
        param_groups = [{}]
        param_groups[0]["lr"] = lr
        param_groups[0]["params"] = []
        for name, param in net.named_parameters():
            param_groups[0]["params"].append(param)

        return param_groups
    else:
        param_groups = [{}, {}]
        param_groups[0]["lr"] = lr / d_tilde
        param_groups[1]["lr"] = lr
        param_groups[0]["params"] = []
        param_groups[1]["params"] = []
        for name, param in net.named_parameters():
            if name.split(".")[-1] == "weight" and len(param.shape) == 2:
                assert param.shape[0] > 0
                if param.shape[0] % ndim == 0 or param.shape[0] == ndim // 2:
                    param_groups[0]["params"].append(param)
                else:
                    param_groups[1]["params"].append(param)
            else:
                param_groups[1]["params"].append(param)

        return param_groups


def group_param_copilot(net, ndim, ndim2, d_tilde, lr, mfm_lora=False):
    if ndim < 1:
        param_groups = [{}]
        param_groups[0]["lr"] = lr
        param_groups[0]["params"] = []
        for name, param in net.named_parameters():
            param_groups[0]["params"].append(param)

        return param_groups, 1.0
    else:
        t = 1.0
        param_groups = [{}]
        param_groups[0]["lr"] = lr / d_tilde
        param_groups[0]["params"] = []

        for name, param in net.named_parameters():
            nl = name.split(".")[0]
            if name.find("dummy") != -1:
                param_groups[0]["params"].append(param)
            elif int(nl) < 40 and int(nl) > 38:
                param_groups[0]["params"].append(param)
            elif mfm_lora and int(nl) <= 38:
                # pass
                param_groups[0]["params"].append(param)

        return param_groups, t


def process_param_groups(
    net, ndim, ndim2=None, d_tilde=1, mode="muT", mfm_lora=False, **kwargs
):
    if mode == "muT":
        param_groups = group_param(net, ndim, d_tilde, kwargs["lr"])
    elif mode == "freezlamma":
        param_groups, t = group_param_copilot(
            net, ndim, ndim2, d_tilde, kwargs["lr"], mfm_lora=mfm_lora
        )

    for param_group in param_groups:
        if "lr" not in param_group:
            param_group["lr"] = kwargs["lr"]
        if "weight_decay" not in param_group:
            param_group["weight_decay"] = kwargs.get("weight_decay", 0.0)

    if mode == "muT":
        return param_groups
    else:
        return param_groups, t


def process_freeze_param(
    net, mode: str = "adaptoronly", lr: float = 1e-5, mfm_lora: bool = False, **kwargs
):
    param_groups = [{}]
    param_groups[0]["lr"] = lr
    param_groups[0]["params"] = []

    if mode == "adaptoronly":
        for name, param in net.named_parameters():
            name.split(".")[0]
            if (
                name.find("adaptor") != -1
                or name.find("dummy") != -1
                # or int(nl) == 0
                or name.find("embed_tokens") != -1
                # or name.find("lm_head") != -1
            ):
                param_groups[0]["params"].append(param)
    else:
        raise Exception("only adaptoronly mode is implemented")

    for param_group in param_groups:
        if "lr" not in param_group:
            param_group["lr"] = kwargs["lr"]
        if "weight_decay" not in param_group:
            param_group["weight_decay"] = kwargs.get("weight_decay", 0.0)

    return param_groups


def myMuAdam(net, impl=Adam, ndim=512, d_tilde=1, **kwargs):
    new_param_groups = []
    for param_group in process_param_groups(net, ndim, d_tilde, **kwargs):
        new_param_groups.extend([param_group])
    return impl(new_param_groups, **kwargs)


def myGroupAdam(
    net, impl=Adam, ndim=512, ndim2=512, d_tilde=1, mfm_lora=False, **kwargs
):
    new_param_groups = []
    param_groups, t = process_param_groups(
        net,
        ndim,
        ndim2=ndim2,
        d_tilde=d_tilde,
        mode="freezlamma",
        mfm_lora=mfm_lora,
        **kwargs,
    )
    for param_group in param_groups:
        new_param_groups.extend([param_group])
    return impl(new_param_groups, **kwargs), t


def myAdam(net, impl=Adam, mode="adaptoronly", mfm_lora=False, **kwargs):
    new_param_groups = []
    param_groups = process_freeze_param(net, mode=mode, mfm_lora=mfm_lora, **kwargs)
    for param_group in param_groups:
        new_param_groups.extend([param_group])
    return impl(new_param_groups, **kwargs)


WARMUP_LOG_RATE = "log"
WARMUP_LINEAR_RATE = "linear"
DECAY_LINEAR_RATE = "linear"
DECAY_COSINE_RATE = "cosine"


def get_torch_optimizer(optimizer):
    if isinstance(optimizer, Optimizer):
        return optimizer

    if hasattr(optimizer, 'optimizer') and isinstance(optimizer.optimizer, Optimizer):
        return optimizer.optimizer

    raise TypeError('{} is not a subclass of torch.optim.Optimizer'.format(type(optimizer).__name__))


class WarmupLR(object):
    """Increase the learning rate of each parameter group from min lr to max lr
        over warmup_num_steps steps, and then fix at max lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_min_lr (float or list): minimum learning rate. Default: 0
            warmup_max_lr (float or list): maximum learning rate. Default: 0.001
            warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
            warmup_type {‘log’, ‘linear’}: increasing function from min_lr to max_lr during warmup. Default: log
            last_batch_iteration (int): The index of the last batch. Default: -1.
        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = WarmupLR(optimizer)
            >>> data_loader = torch.utils.data.DataLoader(...)
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()

    """

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_min_lr: float = 0.0,
                 warmup_max_lr: float = 0.001,
                 warmup_num_steps: int = 1000,
                 warmup_type: str = WARMUP_LOG_RATE,
                 last_batch_iteration: int = -1):

        self.optimizer = get_torch_optimizer(optimizer)

        self.min_lrs = self._format_param(self.optimizer, warmup_min_lr, "min_lr")
        self.max_lrs = self._format_param(self.optimizer, warmup_max_lr, "max_lr")
        self.delta_lrs = [big - small for big, small in zip(self.max_lrs, self.min_lrs)]
        self.warmup_num_steps = max(2, warmup_num_steps)
        # Currently only support linear and log function
        if warmup_type not in {WARMUP_LOG_RATE, WARMUP_LINEAR_RATE}:
            logger.warning(f"Using unknown warmup_type: {warmup_type}. The increasing function "
                           f"is set to default (log)")
            warmup_type = WARMUP_LOG_RATE
        self.warmup_type = warmup_type
        self.inverse_log_warm_up = 1.0 / math.log(self.warmup_num_steps)
        self.last_batch_iteration = last_batch_iteration

    def get_lr(self):
        if self.last_batch_iteration < 0:
            logger.warning("Attempting to get learning rate from scheduler before it has started")
            return [0.0]
        gamma = self._get_gamma()
        return [min_lr + (delta_lr * gamma) for min_lr, delta_lr in zip(self.min_lrs, self.delta_lrs)]

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(self.last_batch_iteration + 1)
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        return 1.0

    def _format_param(self, optimizer, param_value, param_name):
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) != len(optimizer.param_groups):
                raise ValueError("expected {} value for {}, got {}".format(len(optimizer.param_groups), param_name,
                                                                           FileNotFoundError(param_value)))
            return list(param_value)
        return [param_value] * len(optimizer.param_groups)


class groupWarmupDecayLR(WarmupLR):
    """Increase the learning rate of each parameter group from min lr to max lr
    over warmup_num_steps steps, and then decay at linear rate over the remaining training steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_num_steps (int): total number of training steps
        warmup_min_lr (float or list): minimum learning rate. Default: 0
        warmup_max_lr (float or list): maximum learning rate. Default: 0.001
        warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
        warmup_type {'log', 'linear'}: increasing function from min_lr to max_lr during warmup. Default: log
        last_batch_iteration (int): The index of the last batch. Default: -1.
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = WarmupDecayLR(optimizer, 1000000)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_num_steps: int,
        warmup_min_lr: float = 0.0,
        warmup_max_lr: float = 0.001,
        warmup_num_steps: int = 1000,
        warmup_type: str = WARMUP_LINEAR_RATE,
        last_batch_iteration: int = -1,
        d_tilde: float = 1.0,
        decay_type: str = DECAY_LINEAR_RATE,
    ):
        self.total_num_steps = total_num_steps
        super(groupWarmupDecayLR, self).__init__(
            optimizer,
            warmup_min_lr,
            warmup_max_lr,
            warmup_num_steps,
            warmup_type,
            last_batch_iteration,
        )
        self.d_tilde = d_tilde
        self.decay_type = decay_type

        if self.total_num_steps < self.warmup_num_steps:
            logger.warning(
                "total_num_steps {} is less than warmup_num_steps {}".format(
                    total_num_steps, warmup_num_steps
                )
            )
        for group in self.optimizer.param_groups:
            group["lr"] = 0.0

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        # if "d_tilde" in self.optimizer.param_groups[0]:
        #     self.optimizer.param_groups[0]['lr'] *= self.optimizer.param_groups[0]['d_tilde']
        #     self.optimizer.param_groups[1]['lr'] *= self.optimizer.param_groups[1]['d_tilde']
        # else:
        if self.d_tilde >= 1.0:
            self.optimizer.param_groups[0]["lr"] /= self.d_tilde
        elif self.d_tilde < 1.0:
            self.optimizer.param_groups[0]["lr"] *= self.d_tilde
            if len(self.optimizer.param_groups) > 1:
                self.optimizer.param_groups[1]["lr"] *= self.d_tilde

                # self.optimizer.param_groups[0].data._grad *= self.d_tilde

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(
                    self.last_batch_iteration + 1
                )
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        else:
            if self.decay_type == DECAY_LINEAR_RATE:
                return max(
                    0.0,
                    float(self.total_num_steps - self.last_batch_iteration)
                    / float(max(1.0, self.total_num_steps - self.warmup_num_steps)),
                )
            else:
                return 0.5 * (
                    1.0
                    + math.cos(
                        math.pi
                        * float(self.last_batch_iteration - self.warmup_num_steps)
                        / float(max(1.0, self.total_num_steps - self.warmup_num_steps))
                    )
                )
