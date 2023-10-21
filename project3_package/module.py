import torch
import torch.nn as nn
import re

from ast import literal_eval
import warnings


class AttrDict(dict):
    def __init__(self, **kwargs):
        super(AttrDict, self).__init__(**kwargs)
        self.update(kwargs)

    @staticmethod
    def from_dict(dict):
        ad = AttrDict()
        ad.update(dict)
        return ad

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

    def update_from_list(self, str_list: list):
        assert len(str_list) % 2 == 0
        for key, value in zip(str_list[0::2], str_list[1::2]):
            key_list = key.split('.')
            item = None
            last_key = key_list.pop()
            for sub_key in key_list:
                item = self[sub_key] if item is None else item[sub_key]
            try:
                item[last_key] = literal_eval(value)
            except ValueError:
                item[last_key] = value
                warnings.warn('a string value is set to {}'.format(key))


class ConfigurableMixin(object):
    """
    Usage 1: for torch.nn.Module
    >>> import torch.nn as nn
    >>> class Custom(nn.Module, ConfigurableMixin):
    >>>     def __init__(self, config:AttrDict):
    >>>         super(Custom,self).__init__()
    >>>         ConfigurableMixin.__init__(self, config)
    >>>     def forward(self, *input):
    >>>         pass
    >>>     def set_defalut_config(self):
    >>>         self.config.update(dict())
    """

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

    def forward(self, *input):
        raise NotImplementedError

    def set_defalut_config(self):
        raise NotImplementedError('You should set a default config')

    def init_from_weightfile(self):
        if 'weight' not in self.config.GLOBAL:
            return
        if not isinstance(self.config.GLOBAL.weight, dict):
            return
        if 'path' not in self.config.GLOBAL.weight:
            return
        if self.config.GLOBAL.weight.path is None:
            return

        state_dict = torch.load(self.config.GLOBAL.weight.path, map_location=lambda storage, loc: storage)
        ret = {}
        if 'excepts' in self.config.GLOBAL.weight and self.config.GLOBAL.weight.excepts is not None:
            pattern = re.compile(self.config.GLOBAL.weight.excepts)
        else:
            pattern = None

        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k.replace('module.', '')
            if getattr(pattern, 'match', lambda _: False)(k):
                # ignore
                continue
            ret[k] = v

        self.load_state_dict(ret, strict=False)
        _logger.info('Load weights from: {}'.format(self.config.GLOBAL.weight.path))
