The files `emp_psf_f814w_Xx.npy` contain a numpy version of the PSF fits file provided by STScI for F814W filter for WFC3/UVIS `found here <https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/psf>`_.

A few things to note about these PSFs:

- For each PSF, the minimum value has been subtracted out (i.e. the minimum has been set to zero).

- The original PSFs are 4x supersampled with respect to the WFC3/UVIS pixel scale of 0.04 arcseconds. When providing these psfs to paltax you should select the psfs corresponding to the supersampling resolution being used. For example, if the supersampling resolution is set to 1, then the 1x supersampling psf should be used (contained in `emp_psf_f814w_1x.npy`).