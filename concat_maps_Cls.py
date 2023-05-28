from tqdm import tqdm
import h5py as h5
import pickle as pk
import numpy as np

ntot = 3000
lhs_all = np.arange(0, ntot)
nsel_patch = 625
jr = 0
jsnv = 0

jsnv = 0
for jz in [1, 2, 3, 4]:
    maps_all_collage = []
    for jM in tqdm(lhs_all):
        # for runs in runs_all:
        try:
            m1 = pk.load(
                open(
                    './final_files/kappa_tomo_cosmo_' + str(jM) + '_noise_None_SC_False1_noise_' + str(jsnv) + '.pkl',
                    'rb'
                    )
                )
            Om_jM, sig8_jM, w_jM, Ob_jM, h0_jM, ns_jM = m1['config']['Om'], m1['config']['sigma8'], m1['config'][
                'w0'], m1['config']['Ob'], m1['config']['h100'], m1['config']['ns']
            map_ldir = './map_splits/maps_flask_pkdgrav6cosmo_rs_0_ns512_patches_n625_wSN0/'
            df = np.load(map_ldir + '/kappa-gamma-lhs-' + str(jM) + '_jz' + str(jz) + '.npz')
            map_all = df['data']
            if len(maps_all_collage) == 0:
                maps_all_collage = [map_all]
                Om_all = [Om_jM * np.ones(nsel_patch)]
                Ob_all = [Ob_jM * np.ones(nsel_patch)]
                h0_all = [h0_jM * np.ones(nsel_patch)]
                ns_all = [ns_jM * np.ones(nsel_patch)]
                sig8_all = [sig8_jM * np.ones(nsel_patch)]
                w_all = [w_jM * np.ones(nsel_patch)]
            else:
                maps_all_collage.append(map_all)
                Om_all.append(Om_jM * np.ones(nsel_patch))
                Ob_all.append(Ob_jM * np.ones(nsel_patch))
                h0_all.append(h0_jM * np.ones(nsel_patch))
                ns_all.append(ns_jM * np.ones(nsel_patch))
                sig8_all.append(sig8_jM * np.ones(nsel_patch))
                w_all.append(w_jM * np.ones(nsel_patch))

        except:
            pass

    sdir = './'
    print(np.array(maps_all_collage).shape)
    maps_all_collage_mv = np.moveaxis(np.array(maps_all_collage), 3, 0)
    nsim, npatch, nside = maps_all_collage_mv.shape[0], maps_all_collage_mv.shape[1], maps_all_collage_mv.shape[2]
    maps_all_collage_mvf = np.reshape(maps_all_collage_mv, (nsim * npatch, nside, nside))
    np.save(
        sdir + 'processed_data/maps_all_collage_flask_6cosmo_n' + str(nsel_patch) + '_jr' + str(jr) + '_wSN' +
        str(jsnv) + '_jz' + str(jz) + '.npz', maps_all_collage_mvf
        )

    cosmo_parv = np.array(
        [np.array(Om_all),
         np.array(h0_all),
         np.array(ns_all),
         np.array(sig8_all),
         np.array(w_all),
         np.array(Ob_all)]
        )
    cosmo_parv_mv = np.moveaxis(cosmo_parv, 2, 0)
    cosmo_parv_mv = np.moveaxis(cosmo_parv_mv, 2, 1)
    ncosmo = cosmo_parv_mv.shape[2]
    cosmo_parv_mvf = np.reshape(cosmo_parv_mv, (nsim * npatch, ncosmo))
    print(cosmo_parv_mvf.shape, maps_all_collage_mvf.shape)
    np.save(
        sdir + 'processed_data/cosmo_params_all_collage_flask_6cosmo_n' + str(nsel_patch) + '_jr' + str(jr) + '_wSN' +
        str(jsnv) + '_jz' + str(jz) + '.npz', cosmo_parv_mvf
        )
