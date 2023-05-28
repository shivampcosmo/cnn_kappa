# cnn_kappa
1. cosmogrid_analysis_only_kappa.py : This file creates the kappa maps from the pkdgrav simulations at different nside etc
2. save_patches_Cls_namaster_wSN_flask_6cosmo_multz.py: This file adds in the shape noise and then splits up the kappa maps into njk patches. It also measures and saves the power spectra from these patches. The corresponding jackknife patches are provided as well in center_ra_dec_njk%(NJK).txt files
3. concat_maps_Cls.py: This concats all the patches saved from above file and saves the data to be used by CNN regressor. 
4. 
