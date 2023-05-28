import yaml
import pickle
import healpy as hp
import numpy as np
import os
from astropy.table import Table
import gc
# import pyfits as pf
from astropy.io import fits as pf
import timeit
import os
from bornraytrace import lensing as brk
import numpy as np
from bornraytrace import intrinsic_alignments as iaa
import bornraytrace
from astropy.table import Table
import healpy as hp
# import pyfits as pf
from astropy.cosmology import z_at_value
from astropy.cosmology import FlatLambdaCDM, FlatwCDM
from astropy import units as u
import cosmolopy.distance as cd
from scipy.interpolate import interp1d
import gc
import pandas as pd
import pickle
import multiprocessing
from functools import partial


def apply_random_rotation(e1_in, e2_in):
    np.random.seed()  # CRITICAL in multiple processes !
    rot_angle = np.random.rand(len(e1_in)) * 2 * np.pi  #no need for 2?
    cos = np.cos(rot_angle)
    sin = np.sin(rot_angle)
    e1_out = +e1_in * cos + e2_in * sin
    e2_out = -e1_in * sin + e2_in * cos
    return e1_out, e2_out


def IndexToDeclRa(index, nside, nest=False):
    theta, phi = hp.pixelfunc.pix2ang(nside, index, nest=nest)
    return -np.degrees(theta - np.pi / 2.), np.degrees(phi)


def convert_to_pix_coord(ra, dec, nside=1024):
    """
    Converts RA,DEC to hpix coordinates
    """

    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    return pix


def generate_randoms_radec(minra, maxra, mindec, maxdec, Ngen, raoffset=0):
    r = 1.0
    # this z is not redshift!
    zmin = r * np.sin(np.pi * mindec / 180.)
    zmax = r * np.sin(np.pi * maxdec / 180.)
    # parity transform from usual, but let's not worry about that
    phimin = np.pi / 180. * (minra - 180 + raoffset)
    phimax = np.pi / 180. * (maxra - 180 + raoffset)
    # generate ra and dec
    z_coord = np.random.uniform(zmin, zmax, Ngen)  # not redshift!
    phi = np.random.uniform(phimin, phimax, Ngen)
    dec_rad = np.arcsin(z_coord / r)
    # convert to ra and dec
    ra = phi * 180 / np.pi + 180 - raoffset
    dec = dec_rad * 180 / np.pi
    return ra, dec


def addSourceEllipticity(self, es, es_colnames=("e1", "e2"), rs_correction=True, inplace=False):
    """

		:param es: array of intrinsic ellipticities, 

		"""

    #Safety check
    assert len(self) == len(es)

    #Compute complex source ellipticity, shear
    es_c = np.array(es[es_colnames[0]] + es[es_colnames[1]] * 1j)
    g = np.array(self["shear1"] + self["shear2"] * 1j)

    #Shear the intrinsic ellipticity
    e = es_c + g
    if rs_correction:
        e /= (1 + g.conjugate() * es_c)

    #Return
    if inplace:
        self["shear1"] = e.real
        self["shear2"] = e.imag
    else:
        return (e.real, e.imag)


def random_draw_ell_from_w(wi, w, e1, e2):
    '''
    wi: input weights
    w,e1,e2: all the weights and galaxy ellipticities of the catalog.
    e1_,e2_: output ellipticities drawn from w,e1,e2.
    '''

    ell_cont = dict()
    for w_ in np.unique(w):
        mask_ = w == w_
        w__ = np.int(w_ * 10000)
        ell_cont[w__] = [e1[mask_], e2[mask_]]

    e1_ = np.zeros(len(wi))
    e2_ = np.zeros(len(wi))

    for w_ in np.unique(wi):
        mask_ = (wi * 10000).astype(np.int) == np.int(w_ * 10000)
        e1_[mask_] = ell_cont[np.int(w_ * 10000
                                    )][0][np.random.randint(0, len(ell_cont[np.int(w_ * 10000)][0]), len(e1_[mask_]))]
        e2_[mask_] = ell_cont[np.int(w_ * 10000
                                    )][1][np.random.randint(0, len(ell_cont[np.int(w_ * 10000)][0]), len(e1_[mask_]))]

    return e1_, e2_


def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)
        f.close()


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        mute = pickle.load(f)
        f.close()
    return mute


def gk_inv(K, KB, nside, lmax):

    alms = hp.map2alm(K, lmax=lmax, pol=False)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)

    kalmsE = alms / (1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1)))**0.5)

    kalmsE[ell == 0] = 0.0

    alms = hp.map2alm(KB, lmax=lmax, pol=False)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)

    kalmsB = alms / (1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1)))**0.5)

    kalmsB[ell == 0] = 0.0

    _, e1t, e2t = hp.alm2map([kalmsE, kalmsE, kalmsB], nside=nside, lmax=lmax, pol=True)
    return e1t, e2t  # ,r


def g2k_sphere(gamma1, gamma2, mask, nside=1024, lmax=2048, nosh=True):
    """
    Convert shear to convergence on a sphere. In put are all healpix maps.
    """

    gamma1_mask = gamma1 * mask
    gamma2_mask = gamma2 * mask

    KQU_masked_maps = [gamma1_mask, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)
    if nosh:
        almsE = alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1)))**0.5
        almsB = alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1)))**0.5
    else:
        almsE = alms[1] * 1.
        almsB = alms[2] * 1.
    almsE[ell == 0] = 0.0
    almsB[ell == 0] = 0.0
    almsE[ell == 1] = 0.0
    almsB[ell == 1] = 0.0

    almssm = [alms[0], almsE, almsB]

    kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False)
    E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False)
    B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False)

    return E_map, B_map, almsE


def rotate_map_approx(mask, rot_angles, flip=False, nside=2048):
    alpha, delta = hp.pix2ang(nside, np.arange(len(mask)))

    rot = hp.rotator.Rotator(rot=rot_angles, deg=True)
    rot_alpha, rot_delta = rot(alpha, delta)
    if not flip:
        rot_i = hp.ang2pix(nside, rot_alpha, rot_delta)
    else:
        rot_i = hp.ang2pix(nside, np.pi - rot_alpha, rot_delta)
    rot_map = mask * 0.
    rot_map[rot_i] = mask[np.arange(len(mask))]
    return rot_map


def make_maps(seed):
    # st = timeit.default_timer()

    # READ IN PARAMETERS ********************************************

    p, params_dict = seed

    # SET COSMOLOGY ************************************************
    config = dict()
    config['Om'] = params_dict['Omegam']
    config['sigma8'] = params_dict['s8']
    config['ns'] = params_dict['ns']
    config['Ob'] = params_dict['ob']
    config['h100'] = params_dict['h']
    config['w0'] = params_dict['w0']

    config['nside_out'] = 512
    config['nside'] = 512
    config['sources_bins'] = [0, 1, 2, 3]
    config['dz_sources'] = [params_dict['dz1'], params_dict['dz2'], params_dict['dz3'], params_dict['dz4']]
    config['m_sources'] = [params_dict['m1'], params_dict['m2'], params_dict['m3'], params_dict['m4']]
    config['A_IA'] = params_dict['A']
    config['eta_IA'] = params_dict['E']
    config['f'] = params_dict['f']
    config['z0_IA'] = 0.67
    config[
        '2PT_FILE'
        ] = '/global/homes/m/mgatti/Mass_Mapping/HOD/PKDGRAV_CODE//2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_6000HR.fits'

    # cosmology = FlatLambdaCDM(H0=config['h100'] * 100. * u.km / u.s / u.Mpc, Om0=config['Om'])
    cosmology = FlatwCDM(H0=config['h100'] * 100. * u.km / u.s / u.Mpc, Om0=config['Om'], w0=config['w0'])
    '''
    Now the code will try to read in particle counts and make
    kappa,e1,e2 maps. If they are already there, this phase will be skipped
    '''

    # path where kappa/g1/g2 maps are stored
    # base = output_final_maps + '/meta_{0}/'.format(config['f'])
    base_interim = output_intermediate_maps + '/meta_{0}/'.format(config['f'])

    # original path for particle counts
    # shell = np.load(path_sims+'/run_{0}//shells_nside=512.npz'.format(config['f']))
    f_inp = format(config['f'], '06d')
    shell = np.load(path_sims + '/cosmo_' + f_inp + '/run_' + str(params_dict['noise']) + '/shells_nside=512.npz')

    # from ekit import paths as path_tools
    z_bounds = dict()
    z_bounds['z-high'] = np.array([shell['shell_info'][i][3] for i in range(len(shell['shell_info']))])
    z_bounds['z-low'] = np.array([shell['shell_info'][i][2] for i in range(len(shell['shell_info']))])

    i_sprt = np.argsort(z_bounds['z-low'])
    z_bounds['z-low'] = (z_bounds['z-low'])[i_sprt]
    z_bounds['z-high'] = (z_bounds['z-high'])[i_sprt]

    z_bin_edges = np.hstack([z_bounds['z-low'], z_bounds['z-high'][-1]])

    # SAVE LENS MAPS  *****************************************************************
    for s_ in (range(len(z_bounds['z-high']))):
        path_ = base_interim + 'lens_{0}_{1}.fits'.format(s_, config['nside_out'])
        # if not os.path.exists(path_):
        shell_ = shell['shells'][i_sprt[s_]]
        shell_ = (shell_ - np.mean(shell_)) / np.mean(shell_)
        shell_ = hp.ud_grade(shell_, nside_out=config['nside_out'])

        fits_f = Table()
        fits_f['T'] = shell_
        # if os.path.exists(path_):
        # os.remove(path_)
        fits_f.write(path_, overwrite=True)

    # SAVE CONVERGENCE PLANES ********************************************************
    # kappa_pref_evaluated = brk.kappa_prefactor(cosmology.H0, cosmology.Om0, length_unit='Mpc')
    comoving_edges = [cosmology.comoving_distance(x_) for x_ in np.array((z_bounds['z-low']))]

    z_centre = np.empty((len(comoving_edges) - 1))
    for i in range(len(comoving_edges) - 1):
        z_centre[i] = z_at_value(cosmology.comoving_distance, 0.5 * (comoving_edges[i] + comoving_edges[i + 1]))

    un_ = comoving_edges[:(i + 1)][0].unit
    comoving_edges = np.array([c.value for c in comoving_edges])
    comoving_edges = comoving_edges * un_

    overdensity_array = [np.zeros(hp.nside2npix(config['nside_out']))]

    for s_ in (range(len(z_bounds['z-high']))):
        try:
            path_ = base_interim + '/lens_{0}_{1}.fits'.format(s_, config['nside_out'])
            m_ = pf.open(path_)
            overdensity_array.append(m_[1].data['T'])
        except:
            if shell != 0:
                overdensity_array.append(np.zeros(hp.nside2npix(config['nside_out'])))

    overdensity_array = np.array(overdensity_array)
    # print(overdensity_array.shape)

    from bornraytrace import lensing
    kappa_lensing = np.copy(overdensity_array) * 0.

    from tqdm import tqdm
    for i in (np.arange(kappa_lensing.shape[0])):
        try:
            kappa_lensing[i] = lensing.raytrace(
                cosmology.H0,
                cosmology.Om0,
                overdensity_array=overdensity_array[:i].T,
                a_centre=1. / (1. + z_centre[:i]),
                comoving_edges=comoving_edges[:(i + 1)]
                )
        except:
            pass

    mu = pf.open(config['2PT_FILE'])
    redshift_distributions_sources = {'z': None, 'bins': dict()}
    redshift_distributions_sources['z'] = mu[6].data['Z_MID']
    for ix in config['sources_bins']:
        redshift_distributions_sources['bins'][ix] = mu[6].data['BIN{0}'.format(ix + 1)]
    mu = None

    k_tomo = dict()
    k_tomo['config'] = config
    nz_kernel_sample_dict = dict()

    for tomo_bin in config['sources_bins']:
        redshift_distributions_sources['bins'][tomo_bin][250:] = 0.
        nz_sample = brk.recentre_nz(
            np.array(z_bin_edges).astype('float'), redshift_distributions_sources['z'] + config['dz_sources'][tomo_bin],
            redshift_distributions_sources['bins'][tomo_bin]
            )
        nz_kernel_sample_dict[tomo_bin] = nz_sample * (z_bin_edges[1:] - z_bin_edges[:-1])
        k_tomo[tomo_bin] = np.zeros(hp.nside2npix(config['nside_out']))
        kappa_lensing_nz = np.matmul(np.array([nz_kernel_sample_dict[tomo_bin]]), kappa_lensing[:-1, :])[0, :]
        k_tomo[tomo_bin] = kappa_lensing_nz

    save_obj(output_final_maps + p, k_tomo)


# some config
nside = 512  #nside cosmogrid particle count maps
nside_out = 1024  #nside final noisy maps
SC = False  #apply SC or not
noise_rels = 1  # number of noise realisations considered
rot_num = 1  # number of rotations considered (max 4)
A_IA = 0.0
e_IA = 0.0
# runs_cosmo = 1024  # number of cosmogrid independent maps
noise_type = 'None'  # or 'random_depth'

path_sims = '/global/cfs/cdirs/des/cosmogrid/raw/grid/'
import glob
import re

all_subdirs_cosmo = sorted(glob.glob(path_sims + '*/'))
cosmo_i = 0
cosmo_f = 512
all_dirs_here = all_subdirs_cosmo[cosmo_i:cosmo_f]
# Regular expression pattern to extract the number suffix from each string
pattern = r'\d+'

# List comprehension to extract the number suffix as an integer from each string
cosmo_array = [int(re.findall(pattern, s)[0]) for s in all_dirs_here]

output_intermediate_maps = './intermediate_files/'
output_final_maps = './final_files/'
output_temp = './temp_files/'

if not os.path.exists(output_intermediate_maps):
    try:
        os.mkdir(output_intermediate_maps)
    except:
        pass

if __name__ == '__main__':

    import glob
    runstodo = []
    count = 0
    miss = 0

    for f in cosmo_array:
        for i in range(rot_num):

            for nn in range(noise_rels):
                if not os.path.exists(output_intermediate_maps + '/meta_{0}/'.format(f)):
                    try:
                        os.mkdir(output_intermediate_maps + '/meta_{0}/'.format(f))
                    except:
                        pass

                # try:
                f_inp = format(f, '06d')
                with open(path_sims + '/cosmo_' + f_inp + '/run_' + str(nn) + '/params.yml', "r") as f_in:
                    config = yaml.safe_load(f_in.read())

                Omegam = config['Om']
                s8 = config['s8']
                ns = config['ns']
                Ob = config['Ob']
                h = config['H0']
                w0 = config['w0']

                params_dict = dict()
                params_dict['Omegam'] = np.float(Omegam)
                params_dict['s8'] = np.float(s8)

                params_dict['A'] = A_IA
                params_dict['E'] = e_IA
                params_dict['noise'] = nn

                params_dict['rot'] = i
                params_dict['ns'] = np.float(ns)
                params_dict['h'] = np.float(h)
                params_dict['ob'] = np.float(Ob)
                params_dict['w0'] = np.float(w0)
                params_dict['SC'] = SC
                params_dict['f'] = f
                params_dict['m1'] = 0.
                params_dict['m2'] = 0.
                params_dict['m3'] = 0.
                params_dict['m4'] = 0.

                params_dict['dz1'] = 0.
                params_dict['dz2'] = 0.
                params_dict['dz3'] = 0.
                params_dict['dz4'] = 0.

                # p = str(f)+'_noise_'+str(noise_type)+'_SC_'+str(SC)+'_'+str(Omegam )+'_'+str(s8)+'_'+str(ns)+'_'+str(Ob)+'_'+str(h )+'_'+str(A_IA )+'_'+str(e_IA )+'_w'+str(w0)+'_'+str(i+1)+'_noise_'+str(nn)
                p = 'kappa_tomo_cosmo_' + str(f) + '_noise_' + str(noise_type
                                                                  ) + '_SC_' + str(SC) + str(i +
                                                                                             1) + '_noise_' + str(nn)

                if not os.path.exists(output_temp + p + '.pkl'):
                    runstodo.append([p, params_dict])
                    miss += 1
                else:
                    count += 1
                # except:
                #     pass

    print(len(runstodo), count, miss)
    # from tqdm import tqdm
    # for jr in tqdm(range(len(runstodo))):
    #     make_maps(runstodo[jr])

    run_count = 0
    from mpi4py import MPI
    while run_count < len(runstodo):
        comm = MPI.COMM_WORLD
        #
        if (run_count + comm.rank) < len(runstodo):
            #try:
            make_maps(runstodo[run_count + comm.rank])
        #except:
        #    pass
        #if (run_count)<len(runstodo):
        #    make_maps(runstodo[run_count])
        run_count += comm.size
        comm.bcast(run_count, root=0)
        comm.Barrier()

##srun --nodes=4 --tasks-per-node=8 --cpu-bind=cores  python cosmogrid_analysis_only_kappa.py