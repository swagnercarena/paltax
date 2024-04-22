==========================================================================
|logo| paltax
==========================================================================

.. |logo| image:: https://raw.githubusercontent.com/swagnercarena/paltax/main/docs/figures/logo.png
    	:target: https://raw.githubusercontent.com/swagnercarena/paltax/main/docs/figures/logo.png
    	:width: 100

.. |ci| image:: https://github.com/swagnercarena/paltax/workflows/CI/badge.svg
    :target: https://github.com/swagnercarena/paltax/actions

.. |coverage| image:: https://coveralls.io/repos/github/swagnercarena/paltax/badge.svg?branch=main
	:target: https://coveralls.io/github/swagnercarena/paltax?branch=main

.. |license| image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
    :target: https://github.com/swagnercarena/paltax/main/LICENSE

|ci| |coverage| |license|

``paltax`` is a package for conducting simulation-based inference on strong gravitational lensing images.

Installation
------------

``paltax`` is installable via pip:

.. code-block:: bash

    $ pip install paltax

For the most up-to-date version of paltax install directly from the git repository.

.. code-block:: bash

    $ git clone https://github.com/swagnercarena/paltax.git
	$ cd path/to/paltax/
	$ pip install -e .

Usage
-----

The main functionality of ``paltax`` is to train (sequential) neural posterior estimators with on-the-fly data generation. To train a model with ``paltax`` you need a training configuration file that is passed to main.py:

.. code-block:: bash

    $ python main.py --workdir=path/to/model/output/folder --config=path/to/training/configuration

``paltax`` comes preloaded with a number of training configuration files which are described in ``paltax/TrainConfigs/README.rst``. These training configuration files require input configuration files, examples of which can be found in ``paltax``  comes preloaded with a number of configuration files which are described in ``paltax/InputConfigs/``.

Demos
-----

``paltax`` comes with a tutorial notebook for users interested in using the package.

* `Using an input configuration file to generate a batch of images <https://github.com/swagnercarena/paltax/blob/main/notebooks/GenerateImages.ipynb>`_.

Figures
-------

Code for generating the plots included in some of the publications using ``paltax`` can be found under the corresponding arxiv number in the ``notebooks/papers/`` folder.

Attribution
-----------
If you use ``paltax`` for your own research, please cite the ``paltax`` package (`Wagner-Carena et al. 2024 <https://arxiv.org/abs/xxxx.yyyyy>`_)

``paltax`` builds off of the publically released Google DeepMind codebase `jaxstronomy <https://github.com/google-research/google-research/tree/master/jaxstronomy>`_.
