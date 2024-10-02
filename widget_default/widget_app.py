import streamlit as st
import numpy as np

import time
import os
import subprocess
import matplotlib
import altair as alt
import dill
import matplotlib.pyplot as plt

from configurations import *
#from configurations import do_transform_design
from emulator import Trained_Emulators, _Covariance
from bayes_exp import Y_exp_data
#from bayes_plot import obs_tex_labels_2




# https://gist.github.com/beniwohli/765262
greek_alphabet_inv = { u'\u0391': 'Alpha', u'\u0392': 'Beta', u'\u0393': 'Gamma', u'\u0394': 'Delta', u'\u0395': 'Epsilon', u'\u0396': 'Zeta', u'\u0397': 'Eta', u'\u0398': 'Theta', u'\u0399': 'Iota', u'\u039A': 'Kappa', u'\u039B': 'Lamda', u'\u039C': 'Mu', u'\u039D': 'Nu', u'\u039E': 'Xi', u'\u039F': 'Omicron', u'\u03A0': 'Pi', u'\u03A1': 'Rho', u'\u03A3': 'Sigma', u'\u03A4': 'Tau', u'\u03A5': 'Upsilon', u'\u03A6': 'Phi', u'\u03A7': 'Chi', u'\u03A8': 'Psi', u'\u03A9': 'Omega', u'\u03B1': 'alpha', u'\u03B2': 'beta', u'\u03B3': 'gamma', u'\u03B4': 'delta', u'\u03B5': 'epsilon', u'\u03B6': 'zeta', u'\u03B7': 'eta', u'\u03B8': 'theta', u'\u03B9': 'iota', u'\u03BA': 'kappa', u'\u03BB': 'lamda', u'\u03BC': 'mu', u'\u03BD': 'nu', u'\u03BE': 'xi', u'\u03BF': 'omicron', u'\u03C0': 'pi', u'\u03C1': 'rho', u'\u03C3': 'sigma', u'\u03C4': 'tau', u'\u03C5': 'upsilon', u'\u03C6': 'phi', u'\u03C7': 'chi', u'\u03C8': 'psi', u'\u03C9': 'omega', }
greek_alphabet = {v: k for k, v in greek_alphabet_inv.items()}

zeta_over_s_str=greek_alphabet['zeta']+'/s(T)'
eta_over_s_str=greek_alphabet['eta']+'/s(T)'

v2_str='v'u'\u2082'#'{SP}'
v3_str='v'u'\u2083'#'{SP}'
v4_str='v'u'\u2084'#'{SP}'



# short_names = {
#                 'norm' : r'Energy Normalization', #0
#                 'trento_p' : r'TRENTo Reduced Thickness', #1
#                 'sigma_k' : r'Multiplicity Fluctuation', #2
#                 'nucleon_width' : r'Nucleon width [fm]', #3
#                 'dmin3' : r'Min. Distance btw. nucleons cubed [fm^3]', #4
#                 'tau_R' : r'Free-streaming time scale [fm/c]', #5
#                 'alpha' : r'Free-streaming energy dep.', #6
#                 'eta_over_s_T_kink_in_GeV' : r'Temperature of shear kink [GeV]', #7
#                 'eta_over_s_low_T_slope_in_GeV' : r'Low-temp. shear slope [GeV^-1]', #8
#                 'eta_over_s_high_T_slope_in_GeV' : r'High-temp shear slope [GeV^-1]', #9
#                 'eta_over_s_at_kink' : r'Shear viscosity at kink', #10
#                 'zeta_over_s_max' : r'Bulk viscosity max.', #11
#                 'zeta_over_s_T_peak_in_GeV' : r'Temperature of max. bulk viscosity [GeV]', #12
#                 'zeta_over_s_width_in_GeV' : r'Width of bulk viscosity [GeV]', #13
#                 'zeta_over_s_lambda_asymm' : r'Skewness of bulk viscosity', #14
#                 'shear_relax_time_factor' : r'Shear relaxation time normalization', #15
#                 'Tswitch' : 'Particlization temperature [GeV]', #16
# }

short_names = {

       'ylossParam4At2' : '$y_{loss}$ at 2',
       'ylossParam4At4' : '$y_{loss}$ at 4',
       'ylossParam4At6' : '$y_{loss}$ at 6',
       'ylossParam4var': '$y_{loss}$ Var',
       'remnant_energy_loss_fraction' : 'Remnant E loss Frac.',
       'shadowing_factor' : 'Shadowing Factor',
       'tau_form_mean' : r'$\tau_{form}$ Mean',
       'BG' : 'BG',
       'string_source_sigma_x' : 'String Source $\sigma_x$',
       'string_source_sigma_eta': 'String Source $\sigma_\eta$',
       'stringTransverseShiftFrac' : 'String Trans. Shift Frac.',
       'shear_viscosity_3_eta_over_s_T_kink_in_GeV': r'$\frac{\eta}{s}$ $T_{kink}$',
       'shear_viscosity_3_eta_over_s_low_T_slope_in_GeV' : r'$\frac{\eta}{s}$ low-T slope',
       'shear_viscosity_3_eta_over_s_high_T_slope_in_GeV' : r'$\frac{\eta}{s}$ high-T slope',
       'shear_viscosity_3_eta_over_s_at_kink' : r'$\frac{\eta}{s}$ at kink',
       'bulk_viscosity_3_zeta_over_s_max' : r'$\frac{\zeta}{s}$ max',
       'bulk_viscosity_3_zeta_over_s_T_peak_in_GeV' : r'$\frac{\zeta}{s}$ $T_{peak}$',
       'bulk_viscosity_3_zeta_over_s_width_in_GeV' : r'$\frac{\zeta}{s}$ width',
       'bulk_viscosity_3_zeta_over_s_lambda_asymm' : r'$\frac{\zeta}{s}$ $\lambda$ assym.',
       'eps_switch' : 'EPS Switch',
}

system_observables = {
                    'Au-Au-200' : ["dNdeta_eta_cen_00_03_PHOB","dNdeta_eta_cen_10_15_PHOB","dNdeta_eta_cen_25_30_PHOB","dNdeta_eta_cen_40_45_PHOB",
                                   "v22_eta_cen_20_70_STAR","v22_eta_cen_03_15_PHOB","v22_eta_cen_15_25_PHOB","v22_eta_cen_25_50_PHOB",
                                   "v22_pt_cen_00_10_PHEN","v22_pt_cen_20_30_PHEN","v22_pt_cen_30_40_PHEN","v22_pt_cen_50_60_PHEN",
                                   "v22_pt_cen_00_05_STAR","v22_pt_cen_20_30_STAR","v22_pt_cen_30_40_STAR","v22_pt_cen_50_60_STAR",
                                   "meanpT_pi_STAR","meanpT_k_STAR","meanpT_k_PHEN","meanpT_p_PHEN","v22_int_STAR","v32_int_STAR"
                                   ],
                    #'d-Au-200' : ['dNdeta_eta_cen_00_20_PHOB', 'v22_eta_cen_00_05_PHEN', 'v22_pt_cen_00_05_PHEN', 'v32_pt_cen_00_05_PHEN', 'v22_pt_cen_00_10_STAR', 'v32_pt_cen_00_10_STAR']
                    }

obs_lims = {
            #'dNdeta_eta_cen_00_05_frwd_BRAH' : [800, -5.5, 5.5],

            'dNdeta_eta_cen_00_03_PHOB' : [800, -5.5, 5.5],
            'dNdeta_eta_cen_10_15_PHOB' : [800, -5.5, 5.5],
            'dNdeta_eta_cen_25_30_PHOB' : [800, -5.5, 5.5],
            'dNdeta_eta_cen_40_45_PHOB' : [800, -5.5, 5.5],

            'v22_eta_cen_20_70_STAR' : [0.1, -5.5, 5.5],
            'v22_eta_cen_03_15_PHOB' : [0.1, -5.5, 5.5],
            'v22_eta_cen_15_25_PHOB' : [0.1, -5.5, 5.5],
            'v22_eta_cen_25_50_PHOB' : [0.1, -5.5, 5.5],

            'v22_pt_cen_00_10_PHEN' : [0.2, 0, 1.5],
            'v22_pt_cen_20_30_PHEN' : [0.2, 0, 1.5],
            'v22_pt_cen_30_40_PHEN' : [0.2, 0, 1.5],
            'v22_pt_cen_50_60_PHEN' : [0.2, 0, 1.5],

            'v22_pt_cen_00_05_STAR' : [0.2, 0, 1.5],
            'v22_pt_cen_20_30_STAR' : [0.2, 0, 1.5],
            'v22_pt_cen_30_40_STAR' : [0.2, 0, 1.5],
            'v22_pt_cen_50_60_STAR' : [0.2, 0, 1.5],

            'meanpT_pi_STAR' : [1.75, 0, 60],
            'meanpT_k_STAR'  : [1.75, 0, 60],
            'meanpT_k_PHEN'  : [1.75, 0, 60],
            'meanpT_p_PHEN'  : [1.75, 0, 60],

            'v22_int_STAR' : [0.1, 0, 60],
            'v32_int_STAR' : [0.1, 0, 60],

            #'v22_pt_cen_30_40_PHEN' : [0.5, 0, 2],
            #'v32_pt_cen_30_40_PHEN' : [0.2, 0, 2],
            #'v42_pt_cen_30_40_PHEN' : [0.2, 0, 2],
            #'v22_int_STAR' : [.1, 0, 70],
            #'v32_int_STAR' : [.1, 0, 70],
            #'meanpT_pi_PHEN' : [1, 0, 70],
            }


obs_word_labels = {
                    #'dNdeta_eta_cen_00_05_frwd_BRAH' : [r'dN/d'+u'\u03B7   ' + 'large-\u03B7',u'\u03B7'],

                    'dNdeta_eta_cen_00_03_PHOB' : [r'dN/d'+u'\u03B7  0-3% cent. PHOB',u'\u03B7'],
                    'dNdeta_eta_cen_10_15_PHOB' : [r'dN/d'+u'\u03B7  10-15% cent. PHOB',u'\u03B7'],
                    'dNdeta_eta_cen_25_30_PHOB' : [r'dN/d'+u'\u03B7  25-30% cent. PHOB',u'\u03B7'],
                    'dNdeta_eta_cen_40_45_PHOB' : [r'dN/d'+u'\u03B7  40-45% cent. PHOB',u'\u03B7'],

                    'v22_eta_cen_20_70_STAR' : [v2_str+ '(' + u'\u03B7' + ')' + ' 20-70% cent. STAR' ,u'\u03B7'],
                    'v22_eta_cen_03_15_PHOB' : [v2_str+ '(' + u'\u03B7' + ')' + ' 3-15% cent. PHOB',u'\u03B7'],
                    'v22_eta_cen_15_25_PHOB' : [v2_str+ '(' + u'\u03B7' + ')' + ' 15-25% cent. PHOB',u'\u03B7'],
                    'v22_eta_cen_25_50_PHOB' : [v2_str+ '(' + u'\u03B7' + ')' + ' 25-50% cent. PHOB',u'\u03B7'],

                    'v22_pt_cen_00_10_PHEN' : [v2_str+ '(' + 'pT' + ')' + ' 0-10% cent. PHEN', 'pT (GeV)'],
                    'v22_pt_cen_20_30_PHEN' : [v2_str+ '(' + 'pT' + ')' + ' 20-30% cent. PHEN', 'pT (GeV)'],
                    'v22_pt_cen_30_40_PHEN' : [v2_str+ '(' + 'pT' + ')' + ' 30-40% cent. PHEN', 'pT (GeV)'],
                    'v22_pt_cen_50_60_PHEN' : [v2_str+ '(' + 'pT' + ')' + ' 50-60% cent. PHEN', 'pT (GeV)'],

                    'v22_pt_cen_00_05_STAR' : [v2_str+ '(' + 'pT' + ')' + ' 0-5% cent. STAR', 'pT (GeV)'],
                    #'v22_pt_cen_05_10_STAR' : [v2_str+ '(' + 'pT' + ')' + ' 5-10% cent. STAR', 'pT (GeV)'],
                    #'v22_pt_cen_10_20_STAR' : [v2_str+ '(' + 'pT' + ')' + ' 10-20% cent. STAR', 'pT (GeV)'],
                    'v22_pt_cen_20_30_STAR' : [v2_str+ '(' + 'pT' + ')' + ' 20-30% cent. STAR', 'pT (GeV)'],
                    'v22_pt_cen_30_40_STAR' : [v2_str+ '(' + 'pT' + ')' + ' 30-40% cent. STAR', 'pT (GeV)'],
                    #'v22_pt_cen_40_50_STAR' : [v2_str+ '(' + 'pT' + ')' + ' 40-50% cent. STAR', 'pT (GeV)'],
                    'v22_pt_cen_50_60_STAR' : [v2_str+ '(' + 'pT' + ')' + ' 50-60% cent. STAR', 'pT (GeV)'],

                    #'v22_pt_cen_30_40_PHEN' : [v2_str,'transverse momentum (GeV)'],
                    #'v32_pt_cen_30_40_PHEN' : [v3_str,'transverse momentum (GeV)'],
                    #'v42_pt_cen_30_40_PHEN' : [v4_str,'transverse momentum (GeV)'],


                    'meanpT_pi_STAR' : [r'Pion <pT> [GeV]  STAR','Centrality (%)'],
                    'meanpT_k_STAR' :  [r'Kaon <pT> [GeV]  STAR','Centrality (%)'],
                    'meanpT_k_PHEN' :  [r'Kaon <pT> [GeV]  PHEN','Centrality (%)'],
                    'meanpT_p_PHEN' :  [r'Proton <pT> [GeV]  PHEN','Centrality (%)'],

                    'v22_int_STAR' : [v2_str+ ' STAR','Centrality (%)'],
                    'v32_int_STAR' : [v3_str+ ' STAR','Centrality (%)'],



                    #'dNdeta_eta_cen_00_20_PHOB' : r'Charged multiplicity',
                    #'v22_eta_cen_00_05_PHEN' : v2_str,
                    #'v22_pt_cen_00_05_PHEN' : v2_str,
                    #'v32_pt_cen_00_05_PHEN' : v2_str,
                    #'v22_pt_cen_00_10_STAR' : v2_str,
                    #'v32_pt_cen_00_10_STAR' : v3_str,
}


system = 'Au-Au-200'
idf = 0

#@st.cache(persist=True)
def load_design(system):
    #load the design
    design_file = SystemsInfo[system]["main_design_file"]
    range_file = SystemsInfo[system]["main_range_file"]
    design = pd.read_csv(design_file)
    design = design.drop("idx", axis=1)
    labels = design.keys()
    design_range = pd.read_csv(range_file)
    design_max = design_range['max'].values
    design_min = design_range['min'].values
    return design, labels, design_max, design_min



@st.cache_resource(show_spinner=False) #allow_output_mutation=True,
def load_emu(system, idf):
    #load the emulator
    emu = dill.load(open('emulator/emulator-' + system + '-idf-' + str(idf) + "-npc-" + str(SystemsInfo[system]["npc"]) + '.dill', "rb"))
    return emu

@st.cache_data(persist=False)
def load_obs(system):
    observables = system_observables[system]
    nobs = len(observables)
    Yexp = Y_exp_data
    return observables, nobs, Yexp


#@st.cache(allow_output_mutation=True, show_spinner=False)
def emu_predict(emu, params):
    start = time.time()
    Yemu_cov = 0
    #Yemu_mean = emu.predict( np.array( [params] ), return_cov=False )
    Yemu_mean, Yemu_cov = emu.predict( np.array( [params] ), return_cov=True )
    end = time.time()
    time_emu = end - start
    return Yemu_mean, Yemu_cov, time_emu



#@st.cache(show_spinner=False)
def make_plot_altair(observables, Yemu_mean, Yemu_cov, Yexp, idf):
    for iobs, obs in enumerate(observables):
        #print(obs)
        xbins = np.array(obs_cent_list[system][obs])
        #centrality bins
        x = (xbins[:,0]+xbins[:,1])/2.

        #emulator prediction
        y_emu = Yemu_mean[obs][0]
        dy_emu = (np.diagonal(np.abs(Yemu_cov[obs, obs]))**.5)[:,0]
        is_mult = ("dN" in obs) or ("dET" in obs)
        if is_mult and transform_multiplicities:
            dy_emu = np.exp(y_emu+dy_emu) - np.exp(y_emu)
            y_emu = np.exp(y_emu)
        #st.markdown(y_emu)

        #st.markdown(dy_emu)
        df_emu = pd.DataFrame({'cent': x, 'yl':y_emu - dy_emu, "yh":y_emu + dy_emu})

        chart_emu = alt.Chart(df_emu).mark_area(opacity=0.9,color='skyblue').encode(x='cent', y='yl', y2='yh').properties(width=150,height=150)#.color("#ffaa00")



        #experiment
        exp_mean = Yexp[system][obs]['mean']
        exp_err = (Yexp[system][obs]['err'])#[idf])
        #if obs == 'dNdeta_eta_cen_00_03_PHOB':
            #print(exp_err)
        df_exp = pd.DataFrame({"cent": x, obs:exp_mean, obs+"_dy":exp_err, obs+"_dy_low":exp_mean-exp_err, obs+"_dy_high":exp_mean+exp_err})

        # Adjust font size for the v_n's
        normal_font_size=14
        if (obs in ['v22','v32','v42']):
            normal_font_size=18

        pre_chart_exp=alt.Chart(df_exp)

        chart_exp = pre_chart_exp.mark_circle(color='Black',size=10).encode(
        x=alt.X( 'cent', axis=alt.Axis(title=obs_word_labels[obs][1], titleFontSize=14), scale=alt.Scale(domain=(obs_lims[obs][1], obs_lims[obs][2])) ),
        y=alt.Y(obs, axis=alt.Axis(title=obs_word_labels[obs][0], titleFontSize=normal_font_size), scale=alt.Scale(domain=(0, obs_lims[obs][0]))  )
        )

        # generate the error bars
        errorbars = pre_chart_exp.mark_errorbar().encode(
                x=alt.X('cent', axis=alt.Axis(title='')),
                y=alt.Y(obs+"_dy_low", axis=alt.Axis(title=''), scale=alt.Scale(domain=(0, obs_lims[obs][0]))  ),
                y2=alt.Y2(obs+"_dy_high"),
        )

        chart = alt.layer(chart_emu, chart_exp + errorbars)

        if iobs == 0:
            charts0 = chart
        if iobs in [1, 2, 3]:
            charts0 = alt.hconcat(charts0, chart)

        if iobs == 4:
            charts1 = chart
        if iobs in [5, 6, 7]:
            charts1 = alt.hconcat(charts1, chart)

        if iobs == 8:
            charts2 = chart
        if iobs in [9, 10, 11]:
            charts2 = alt.hconcat(charts2, chart)

        if iobs == 12:
            charts3 = chart
        if iobs in [13, 14, 15]:
            charts3 = alt.hconcat(charts3, chart)

        if iobs == 16:
            charts4 = chart
        if iobs in [17, 18, 19]:
            charts4 = alt.hconcat(charts4, chart)

        if iobs == 20:
            charts5 = chart
        if iobs in [21]:
            charts5 = alt.hconcat(charts5, chart)

    charts0 = st.altair_chart(charts0)
    charts1 = st.altair_chart(charts1)
    charts2 = st.altair_chart(charts2)
    charts3 = st.altair_chart(charts3)
    charts4 = st.altair_chart(charts4)
    charts5 = st.altair_chart(charts5)

    #return charts0, charts1, charts2


# # MAP
# params_0 = [ 1.88390055e-01,  3.17747161e+00,  1.89067202e+00,  9.60504520e-01,
#   1.00001452e-03,  4.79297981e-01,  2.00001363e-01,  2.12701486e+01,
#   4.32353479e-01,  7.99999976e-01,  9.99991586e-01,  2.29995734e-01,
#  -1.99999883e+00, -8.96047350e-01,  1.20491669e-02,  1.51026518e-01,
#   2.47844977e-01,  3.53000756e-02, -7.99999976e-01,  5.74998416e-01]
# params= []

# #updated params
# for i_s, s_name in enumerate(short_names.keys()):
#     min = 0. #design_min[i_s]
#     max = 100. #design_max[i_s]
#     step = (max - min)/100.
#     p = st.sidebar.slider(short_names[s_name], min_value=min, max_value=max, value=params_0[i_s], step=step)
#     params.append(p)




def main():


    st.write('Based on widget written by Derek Everett for the 2D JETSCAPE calibration')
    st.title('Hadronic Bulk Observable Emulator for Top RHIC Energy')
    #st.markdown('Our [model](https://inspirehep.net/literature/1821941) for the outcome of [ultrarelativistic heavy ion collisions](https://home.cern/science/physics/heavy-ions-and-quark-gluon-plasma) include many parameters which affects final hadronic observables in non-trivial ways. You can see how each observable (blue band) depends on the parameters by varying them using the sliders in the sidebar(left). All observables are plotted as a function of centrality for Pb nuclei collisions at'r'$\sqrt{s_{NN}} = 2.76$ TeV.')
    st.markdown('Our 3D model describing the dynamics of ultrarelativistic nuclear collisions includes 20 parameters which affect final hadronic observables in non-trivial ways. You can explore how each observable (blue band) depends on the parameters by varying them using the sliders in the sidebar (left). Observables are plotted as a function of pseudo-rapidity, transverse momentum, or centrality for Au-Au or d-Au collisions at'r'$\sqrt{s_{NN}} = 200$ GeV.')
    st.markdown('Experimentally measured observables by four RHIC collaborations are shown with black markers.')
    st.markdown('The bottom row of figures displays the temperature dependence of the specific shear and bulk viscosities (red lines), as determined by corresponding parameters on the left sidebar.')
    st.markdown('By default, the 20 parameters are assigned the values that *best* fit the experimental data (maximize the likelihood).')
    #st.markdown(r'An important modelling ingredient is the particlization model used to convert hydrodynamic fields into individual hadrons. Three different viscous correction models can be selected by clicking the "Particlization model" button below.')


    # Reset button
    st.markdown('<a href="javascript:window.location.href=window.location.href">Reset</a>', unsafe_allow_html=True)


    # load the design
    design, labels, design_max, design_min = load_design(system)

    # load the emu
    emu = load_emu(system, idf=0)

    # load the exp obs
    observables, nobs, Yexp = load_obs(system)


        # initialize parameters
    # # old MAP
    # params_0 = [ 9.60009249e-01,  9.60009254e-01,  2.33078395e+00,  9.99999991e-01,
    # 4.79338046e-01,  9.80067247e-03,  5.09907229e-01,  7.40687126e+00,
    # 4.09929837e-01,  2.90269144e-01,  1.00038845e-03,  1.51111083e-01,
    # -1.99999704e+00, -1.67523375e-01,  5.14814374e-02,  1.99569809e-01,
    # 2.15126190e-01,  1.01055139e-01,  5.99999956e-01,  4.70604189e-01]

    # # MAP with only v2pt & v2eta
    # params_0 = [ 1.12532695e-01,  1.12534255e-01,  2.85410817e+00,  1.00007840e-03,
    # 8.50654497e-01,  9.99999910e-01,  2.00000017e-01,  2.49999684e+01,
    # 2.49581744e-01,  3.74219869e-01,  8.38397357e-01,  1.82222569e-01,
    # 9.99997140e-01, -9.99987730e-01,  3.11108393e-02,  1.99999564e-01,
    # 2.99999900e-01,  2.50001219e-02,  5.99996781e-01,  1.00000026e-01]


    # default MAP (with PID)
    params_0 = [ 2.01911982e-03,  1.35910129e+00,  2.80975673e+00,  1.00245072e-03,
    5.82803457e-01,  1.01121899e-03,  9.14306347e-01,  2.49999682e+01,
    3.45655826e-01,  1.98789661e-01,  9.99983158e-01,  2.13333685e-01,
    9.99999388e-01,  6.70209547e-02,  1.40791956e-01,  3.16532291e-02,
    2.25685666e-01,  2.50001497e-02, -7.99989754e-01,  1.53310209e-01]


    #params_0 = MAP_params[system][ idf_label_short[idf] ]
    params = []

    #updated params
    for i_s, s_name in enumerate(short_names.keys()):
        min = design_min[i_s]
        max = design_max[i_s]
        step = (max - min)/100.
        p = st.sidebar.slider(short_names[s_name], min_value=min, max_value=max, value=params_0[i_s], step=step)
        params.append(p)


    # get emu prediction
    Yemu_mean, Yemu_cov, time_emu = emu_predict(emu,params)

    #redraw plots
    make_plot_altair(observables, Yemu_mean, Yemu_cov, Yexp, idf)
    #make_plot_eta_zeta(params)

    st.header('How it works')
    st.markdown('A description of the physics model and parameters can be found [here](https://arxiv.org/pdf/2203.04685).')
    st.markdown('The observables above (and additional ones not shown) are combined into [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) (PC).')
    st.markdown('A [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) (GP) is fitted to each of the dominant principal components by running our physics model on a coarse [space-filling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) set of points in parameter space.')
    st.markdown('The Gaussian Process is then able to interpolate between these points, while estimating its own uncertainty.')







if __name__ == "__main__":
    main()
