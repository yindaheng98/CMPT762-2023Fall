import torch.nn as nn


class AttrDict(dict):
    def __init__(self, **kwargs):
        super(AttrDict, self).__init__(**kwargs)
        self.update(kwargs)

    def __setitem__(self, key: str, value):
        super(AttrDict, self).__setitem__(key, value)
        super(AttrDict, self).__setattr__(key, value)

    def update(self, config: dict):
        for k, v in config.items():
            if k not in self:
                self[k] = AttrDict()
            if isinstance(v, dict):
                self[k].update(v)
            else:
                self[k] = v


class ConfigurableMixin(object):
    def __init__(self, config):
        self._cfg = AttrDict(

        )
        self.set_defalut_config()
        self._cfg.update(config)

    def set_defalut_config(self):
        raise NotImplementedError

    @property
    def config(self):
        return self._cfg


class CVModule(nn.Module, ConfigurableMixin):
    __Keys__ = ['GLOBAL', ]

    def __init__(self, config):
        super(CVModule, self).__init__()
        ConfigurableMixin.__init__(self, config)
        for key in CVModule.__Keys__:
            if key not in self.config:
                self.config[key] = dict()
