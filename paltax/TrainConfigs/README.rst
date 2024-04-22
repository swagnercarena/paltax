Config Files
------------

NPE Configs
-----------

train_config_npe_base.py
    Base configuration for training the NPE model. Equivalent to the Fiducial model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_50k.py
    Base NPE where unique images are limited to 50 thousand images (random rotations are applied at the simulation level). Equivalent to the :math:`5 \times 10^{4}` model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_50k.py
    Base NPE where unique images are limited to 500 thousand images (random rotations are applied at the simulation level). Equivalent to the :math:`5 \times 10^{5}` model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_5M.py
    Base NPE where unique images are limited to 5 million images (random rotations are applied at the simulation level). Equivalent to the :math:`5 \times 10^{6}` model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_constant.py
    Base NPE with constant learning rate. Equivalent to the Constant Learning Rate model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_exp_fast.py
    Base NPE with exponential learning rate with decay factor 0.98. Equivalent to the Exponential Decay: 0.98 model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_exp_slow.py
    Base NPE with exponential learning rate with decay factor 0.99. Equivalent to the Exponential Decay: 0.99 model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_linear.py
    Base NPE with linear learning rate. Equivalent to the Linear Decay, Learning Rate: :math:`10^{-2}` model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_linear_0p001.py
    Base NPE with linear learning rate and base learning rate modified to :math:`10^{-3}`. Equivalent to the Linear Decay, Learning Rate: :math:`10^{-3}` model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_res_18_very_small.py
    Base NPE config with ``ResNet18VerySmall`` model in place of ``ResNet50``. Equivalent to the Resnet 18 Very Small model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_res_18_small.py
    Base NPE config with ``ResNet18Small`` model in place of ``ResNet50``. Equivalent to the Resnet 18 Small model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_res_18.py
    Base NPE config with ``ResNet18`` model in place of ``ResNet50``. Equivalent to the Resnet 18 model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_res_34.py
    Base NPE config with ``ResNet34`` model in place of ``ResNet50``. Equivalent to the Resnet 34 model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_res_d.py
    Base NPE config with ``ResNetD50`` model in place of ``ResNet50``. Equivalent to the Resnet-D 50 model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.

SNPE Configs
-----------

train_config_snpe_base.py
    Base configuration for training the SNPE model. Equivalent to the Sequential model in `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_snpe_instant_avg.py
    Base SNPE config with proposal updated using averaging of current proposal and prior. Equivalent to the Modified Proposal model in Appendix D of `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.
train_config_snpe_m_ad.py
    Base SNPE config with a later start to sequential (200 epochs). Equivalent to the Late Start Sequential model in Appendix D of `Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_.