import numpy as np
import healpy as hp
import scipy as sp
import pymaster as nmt
import scipy
from scipy.interpolate import RegularGridInterpolator
import sys, os
from tqdm import tqdm
import pickle as pk


def get_cutout(jM, snv, jsnv, jz, jr, saveinit=False):
    # nsamp = 1024
    # nsamp = 128
    # njk = 150
    # njk = 2500
    njk = 625
    xorig = 66
    xsize_f = int(xorig * np.sqrt(625 / njk))

    # create directory if does not exist:
    # if saveinit:
    sdir_Cls = './Cl_splits/Cls_pkdgrav6cosmo_rs_' + str(jr) + '_ns512_patches_n' + str(njk) + '_wSN' + str(jsnv)
    sdir_maps = './map_splits/maps_pkdgrav6cosmo_rs_' + str(jr) + '_ns512_patches_n' + str(njk) + '_wSN' + str(jsnv)
    if not os.path.exists(sdir_Cls):
        os.makedirs(sdir_Cls)
    if not os.path.exists(sdir_maps):
        os.makedirs(sdir_maps)

    print('jM is ', jM)
    all_cens = np.loadtxt('center_ra_dec_njk' + str(njk) + '.txt').T
    try:
        m1 = pk.load(
            open(
                './final_files/kappa_tomo_cosmo_' + str(jM) + '_noise_None_SC_False1_noise_' + str(jsnv) + '.pkl', 'rb'
                )
            )
        have_file = 1
    except:
        have_file = 0
        pass
    if have_file:
        m1 = m1[jz]
        m1 -= np.mean(m1)

        size_deg = hp.nside2resol(hp.npix2nside(len(m1)), arcmin=True) * xsize_f / 60.
        Cls_all = []
        map_all = []
        for jc in (range(len(all_cens))):
            raj, decj = all_cens[jc][0], all_cens[jc][1]
            mt1 = hp.gnomview(
                m1,
                rot=(raj, decj),
                xsize=xsize_f,
                no_plot=True,
                reso=hp.nside2resol(hp.npix2nside(len(m1)), arcmin=True),
                return_projected_map=True
                )
            if snv > 0:
                noise_map = np.random.normal(0.0, snv, (mt1.shape))
            else:
                noise_map = 0.0
            mt1 += noise_map

            Lx = size_deg * np.pi / 180.
            Nx = mt1.shape[0]
            Ly = Lx

            maskf = np.ones_like(mt1)
            f0 = nmt.NmtFieldFlat(Lx, Ly, maskf, [mt1])
            w00 = nmt.NmtWorkspaceFlat()

            l0_bins = np.arange(Nx / 8) * 8 * np.pi / Lx
            lf_bins = (np.arange(Nx / 8) + 1) * 8 * np.pi / Lx
            bf = nmt.NmtBinFlat(l0_bins, lf_bins)
            ells_uncoupled = bf.get_effective_ells()

            if saveinit:
                # print(jc, jM, jsnv)
                # if jc == 0 and jM == 0 and jsnv == 0:
                np.save(sdir_Cls + '/ells_kappa', ells_uncoupled)
                w00 = nmt.NmtWorkspaceFlat()
                w00.compute_coupling_matrix(f0, f0, bf)
                w00.write_to(sdir_Cls + '/w00_flat.fits')
                return 0

            w00.read_from(sdir_Cls + '/w00_flat.fits')
            cl00_coupled = nmt.compute_coupled_cell_flat(f0, f0, bf)
            cl00_uncoupled = w00.decouple_cell(cl00_coupled)

            if len(Cls_all) == 0:
                Cls_all = cl00_uncoupled
            else:
                Cls_all = np.vstack((Cls_all, cl00_uncoupled))

            if len(map_all) == 0:
                map_all = mt1.reshape(xsize_f, xsize_f, 1)
            else:
                map_all = np.dstack((map_all, mt1.reshape(xsize_f, xsize_f, 1)))

        np.save(sdir_Cls + '/Cls_kappa-lhs-' + str(jM) + '_jz' + str(jz), Cls_all)
        print('saving at:', sdir_maps + '/kappa-gamma-lhs-' + str(jM) + '_jz' + str(jz) + '.npz')
        np.savez_compressed(
            sdir_maps + '/kappa-gamma-lhs-' + str(jM) + '_jz' + str(jz) + '.npz', data=map_all.data, mask=map_all.mask
            )

    return 0


def save_Cls_batch(jrank, njobs, snv, jsnv):
    ni = 0
    nf = 2700
    lhs_all = np.arange(ni, nf)
    lhs_all_split = np.array_split(lhs_all, njobs)
    lhs_jrank = lhs_all_split[jrank]

    # for lhs in tqdm(lhs_jrank):
    for lhs in (lhs_jrank):
        get_cutout(lhs, snv, jsnv, 0, 0)
        get_cutout(lhs, snv, jsnv, 1, 0)
        get_cutout(lhs, snv, jsnv, 2, 0)
        get_cutout(lhs, snv, jsnv, 3, 0)


from mpi4py import MPI
if __name__ == '__main__':
    run_count = 0
    n_jobs = 16
    sigmae = 0.314
    smoothing = hp.nside2resol(512, arcmin=True)
    neff = 1.461

    snv = 0.0
    jsnv = 0

    # If you want to put in DES shape noise as well here:
    # snv = sigmae/np.sqrt(neff * (smoothing**2))
    # jsnv = 1
    # Scale with some factor for lower shape noise.

    while run_count < n_jobs:
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if (run_count + comm.rank) < n_jobs:
            save_Cls_batch(comm.rank, n_jobs, snv, jsnv)
        run_count += comm.size
        comm.bcast(run_count, root=0)
        comm.Barrier()

# # salloc -N 4 -C haswell -q interactive -t 04:00:00 -L SCRATCH
# # srun --nodes=4 --tasks-per-node=20 --cpu-bind=cores python save_patches_Cls_namaster_wSN_flask_6cosmo_multz.py
