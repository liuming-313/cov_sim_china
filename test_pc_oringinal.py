# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 16:32:33 2022

@author: m1550
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import covasim as cv
import sciris as sc
import pylab as pl
import pandas as pd
import datetime as dt
import matplotlib.ticker as ticker

test_para = [0.3]
test_acu = 64
para_data = pd.read_excel(r'D:\onedrive\OneDrive - HKUST Connect\Desktop\IA\OneDrive - HKUST\PHD\ming\paras.xlsx',
                          header=1)
# para_data = pd.read_excel(r'paras.xlsx', header=1)
vac_data = pd.read_excel(r'D:\onedrive\OneDrive - HKUST Connect\Desktop\IA\OneDrive - HKUST\PHD\ming\feature.xlsx',
                         sheet_name='vac_data')
# vac_data = pd.read_excel(r'feature.xlsx', sheet_name='vac_data')


cv.check_version('3.1.2')

##input para:
do_plot = 1
do_save = 1
save_sim = 1
plot_hist = 0  # whether to keep people
sims_cutoff = 9999
to_plot = sc.objdict({
    'Daily infections': ['new_infections'],
    'Cumulative infections': ['cum_infections'],
    'Daily hospitalisations': ['new_severe'],
    'Occupancy of hospitalisations': ['n_severe'],  # 'Cumulative hospitalisations': ['cum_severe']
    'Daily ICUs': ['new_critical'],
    'Occupancy of ICUs': ['n_critical'],
    'Cumulative hospitalisations': ['cum_severe'],
    'Cumulative ICUs': ['cum_critical'],
    # 'Cumulative ICUs': ['cum_critical'],
    # 'Daily quarantined':['new_quarantined'],
    # 'Number in quarantined':['n_quarantined'],
    'Daily deaths': ['new_deaths'],
    'Cumulative deaths': ['cum_deaths'],
    # 'R': ['r_eff'],
    # 'number vaccinated': ['n_vaccinated'],
    # 'proportion vaccinated': ['frac_vaccinated'],
    # 'Vaccinations ': ['cum_vaccinated'],
})
age_trace_time = ['2022-03-15', '2022-03-23',
                  '2022-03-31', '2022-04-08', '2022-04-15', '2022-04-30',
                  '2022-05-15', '2022-06-01']  # trace the age-distribution data
sum_trace_time = ['2022-03-15', '2022-03-23',
                  '2022-03-31', '2022-04-08', '2022-04-15', '2022-04-30',
                  '2022-05-15', '2022-06-01']  # trace the new and accumulative cases
trace_state = ['infectious', 'severe', 'critical', 'dead']


# vac_source_data={'80%_sp_s':[0.07877133120554,0.0148604083460425,0.0518208994033026,0.106019344967553,0.173855278895777,0.213384390258624,0.194904582585975,0.0908922012810169,0.0754915630561687,2403842.39267718],'80%_sp_b':[0.000255942125232749,0.0825690600506097,0.168966060200137,0.21739870044071,0.197390494660928,0.161254010606902,0.109120474259607,0.0401874102331159,0.0228578474227573,3407546.60732282],'80%_nm_s':[0.022684858933556,0.0164742223274809,0.0569085420392444,0.116328357836889,0.190179618627885,0.233911381831773,0.214980054080038,0.101349908925614,0.0471830553975205,2284989.19511955],'80%_nm_b':[6.77548396742691E-05,0.0841438098561791,0.170569997433851,0.21927447330802,0.19848741588474,0.16249121475172,0.110640215869047,0.0411924301124197,0.0131326879443496,3523622.00488045],'90%_sp_s':[0.131116276801493,0.013798771571868,0.047319906504668,0.0966633705350033,0.153880653207854,0.19422543719918,0.179366764461889,0.0852671536561561,0.0983616660618887,2845281.60545462],'90%_sp_b':[0.000463573314478996,0.0834286574943991,0.16789070077091,0.215686018709344,0.190112471212468,0.159713717707448,0.109273386167382,0.041023564872425,0.0324079097511442,3706575.39454538],'90%_nm_s':[0.0524918381549326,0.0173431903118315,0.0541308542672054,0.108489349373299,0.164762316561041,0.213334876261318,0.22240792641718,0.108800735953812,0.0582389126993804,2657366.08434635],'90%_nm_b':[0.00016494116674415,0.0931921392717168,0.170688045974821,0.215140724093622,0.180908901763672,0.155909806765904,0.120419868843402,0.0465220588117879,0.0170535133083295,3895143.91565365],'80%_sp_sboost':[0.07877133120554,0.0148604083460425,0.0518208994033026,0.106019344967553,0.173855278895777,0.213384390258624,0.194904582585975,0.0908922012810169,0.0754915630561687,961536.957070873],'80%_sp_bboost':[0.000255942125232749,0.0825690600506097,0.168966060200137,0.21739870044071,0.197390494660928,0.161254010606902,0.109120474259607,0.0401874102331159,0.0228578474227573,1363018.64292913],'80%_nm_sboost':[0.022684858933556,0.0164742223274809,0.0569085420392444,0.116328357836889,0.190179618627885,0.233911381831773,0.214980054080038,0.101349908925614,0.0471830553975205,913995.67804782],'80%_nm_bboost':[6.77548396742691E-05,0.0841438098561791,0.170569997433851,0.21927447330802,0.19848741588474,0.16249121475172,0.110640215869047,0.0411924301124197,0.0131326879443496,1409448.80195218],'90%_sp_sboost':[0.131116276801493,0.013798771571868,0.047319906504668,0.0966633705350033,0.153880653207854,0.19422543719918,0.179366764461889,0.0852671536561561,0.0983616660618887,1422640.80272731],'90%_sp_bboost':[0.000463573314478996,0.0834286574943991,0.16789070077091,0.215686018709344,0.190112471212468,0.159713717707448,0.109273386167382,0.041023564872425,0.0324079097511442,1853287.69727269],'90%_nm_sboost':[0.0524918381549326,0.0173431903118315,0.0541308542672054,0.108489349373299,0.164762316561041,0.213334876261318,0.22240792641718,0.108800735953812,0.0582389126993804,1328683.04217318],'90%_nm_bboost':[0.00016494116674415,0.0931921392717168,0.170688045974821,0.215140724093622,0.180908901763672,0.155909806765904,0.120419868843402,0.0465220588117879,0.0170535133083295,1947571.95782682],'base__s':[0.0028221345335779,0.0165948509667998,0.0586809008233434,0.1202039335906,0.197988860442573,0.24226712640713,0.219292601608092,0.100618469908147,0.0415311217197373,2000259],'base__b':[8.20369134557892E-06,0.0824931649052395,0.17117853914344,0.220520271794605,0.201111600177326,0.163794901405829,0.109841432112088,0.039801470669437,0.0112504161006908,3169305],'base__sboost':[2.13076679740833E-05,0.00452050371327091,0.0526430523070468,0.111499748405613,0.221644001317797,0.266308151494241,0.227098764319163,0.0943733005085976,0.0218911702662967,610109],'base__bboost':[9.7814338603896E-07,0.00980295301488245,0.0865500393702713,0.181424078955734,0.250598379216409,0.235284566364583,0.167027764600013,0.0581408428661557,0.0111703974685649,1022345]}
# vac_data=pd.DataFrame(vac_source_data)
def check(sim):
    age_3_12 = (sim.people.age >= 3) * (
            sim.people.age < 12)  # cv.true() returns indices of people matching this condition, i.e. people under 20
    age_12_19 = (sim.people.age >= 12) * (sim.people.age < 20)
    age_20_29 = (sim.people.age >= 20) * (sim.people.age < 30)  # Multiplication means "and" here
    age_30_39 = (sim.people.age >= 30) * (sim.people.age < 40)  # Multiplication means "and" here
    age_40_49 = (sim.people.age >= 40) * (sim.people.age < 50)  # Multiplication means "and" here
    age_50_59 = (sim.people.age >= 50) * (sim.people.age < 60)  # Multiplication means "and" here
    age_60_69 = (sim.people.age >= 60) * (sim.people.age < 70)  # Multiplication means "and" here
    age_70_79 = (sim.people.age >= 70) * (sim.people.age < 80)  # Multiplication means "and" here
    age_80 = sim.people.age >= 80
    age_3_12_severe = age_3_12


def make_sim(n_beds_hosp=2000, n_beds_icu=255, ful_vac_rate=0.692, third_vac_rate='', soc_dis_rate=0.7, factor=100,
             method='', ):
    '''
    :param n_beds_hosp: number of islation beds in hospital
    :param n_beds_icu:  number of ICU beds
    :param ful_vac_rate: 80%,90%
    :param soc_dis_rate: default: 0.7
    :param factor:
    :param method: nm: normal ; sp:spcial for elder and child
    :return: sim
    '''

    def protect_reinfection(sim):
        sim.people.rel_sus[sim.people.recovered] = 0.0

    total_pop = 7.462e6  # HK polulation size
    pop_size = int(7.462e6 / factor)
    pop_scale = factor
    pop_type = 'hybrid'
    beta = 0.016  # previous value: 0.016
    verbose = 0
    seed = 1
    asymp_factor = 1  # multiply beta by this factor for asymptomatic cases; no statistically significant difference in transmissibility
    pars = sc.objdict(
        use_waning=True,
        pop_size=pop_size,
        pop_infected=0,
        pop_scale=pop_scale,
        pop_type=pop_type,
        beta=beta,
        asymp_factor=asymp_factor,
        rescale=True,
        rand_seed=seed,
        verbose=verbose,
        nab_boost=6,
        start_day='2021-12-15',
        end_day='2022-09-30',

        dur={'exp2inf': {'dist': 'lognormal_int', 'par1': 4.0, 'par2': 1.0},
             'inf2sym': {'dist': 'lognormal_int', 'par1': 0.25, 'par2': 0.2},
             'sym2sev': {'dist': 'lognormal_int', 'par1': 0.5, 'par2': 0.5},
             'sev2crit': {'dist': 'lognormal_int', 'par1': 0.5, 'par2': 0.5},
             'asym2rec': {'dist': 'lognormal_int', 'par1': 5.0, 'par2': 2.0},
             'mild2rec': {'dist': 'lognormal_int', 'par1': 5.0, 'par2': 2.0},
             'sev2rec': {'dist': 'lognormal_int', 'par1': 8.0, 'par2': 2.0},
             'crit2rec': {'dist': 'lognormal_int', 'par1': 15, 'par2': 6.3},
             'crit2die': {'dist': 'lognormal_int', 'par1': 0.5, 'par2': 0}},
        ##severity parameters
        n_beds_hosp=int(n_beds_hosp / factor),
        n_beds_icu=int(n_beds_icu / factor),
        no_hosp_factor=2,
        no_icu_factor=2,  # without beds critically cases will 2 times likely to die
    )

    sim = cv.Sim(pars=pars, location='china, hong kong special administrative region',
                 analyzers=[cv.age_histogram(days=age_trace_time, states=trace_state),
                            cv.daily_age_stats(states=trace_state)])  # use age-distribution data
    rel_beta = 7.018
    rel_severe_prob = 0.4
    rel_crit_prob = 0.36
    rel_death_prob = 0.8
    omicron1 = cv.variant('delta', days=sim.day('2022-01-15'), n_imports=16)  # np.array([16,15,125,98,97])
    omicron2 = cv.variant('delta', days=sim.day('2022-01-16'), n_imports=15)
    omicron3 = cv.variant('delta', days=sim.day('2022-01-17'), n_imports=125)
    omicron4 = cv.variant('delta', days=sim.day('2022-01-18'), n_imports=98)
    omicron5 = cv.variant('delta', days=sim.day('2022-01-19'), n_imports=97)
    omicron1.p['rel_beta'] = rel_beta
    omicron1.p['rel_severe_prob'] = rel_severe_prob  ##1/7*0.47*3.2
    omicron1.p['rel_crit_prob'] = rel_crit_prob
    omicron1.p['rel_death_prob'] = rel_death_prob
    omicron2.p['rel_beta'] = rel_beta
    omicron2.p['rel_severe_prob'] = rel_severe_prob  ##1/7*0.47*3.2
    omicron2.p['rel_crit_prob'] = rel_crit_prob
    omicron2.p['rel_death_prob'] = rel_death_prob
    omicron3.p['rel_beta'] = rel_beta
    omicron3.p['rel_severe_prob'] = rel_severe_prob  ##1/7*0.47*3.2
    omicron3.p['rel_crit_prob'] = rel_crit_prob
    omicron3.p['rel_death_prob'] = rel_death_prob
    omicron4.p['rel_beta'] = rel_beta
    omicron4.p['rel_severe_prob'] = rel_severe_prob  ##1/7*0.47*3.2
    omicron4.p['rel_crit_prob'] = rel_crit_prob
    omicron4.p['rel_death_prob'] = rel_death_prob
    omicron5.p['rel_beta'] = rel_beta
    omicron5.p['rel_severe_prob'] = rel_severe_prob  ##1/7*0.47*3.2
    omicron5.p['rel_crit_prob'] = rel_crit_prob
    omicron5.p['rel_death_prob'] = rel_death_prob
    sim['variants'] += [omicron1, omicron2, omicron3, omicron4, omicron5]

    if ful_vac_rate == 0.8:
        nu = '80%'
    elif ful_vac_rate == 0.9:
        nu = '90%'
    else:
        nu = 'base'

    def vaccinate_by_age_Sinovac(sim):
        key = nu + '_' + method + '_s'
        key_r = nu + '_' + method + '_r'
        age_3_12 = cv.true((sim.people.age >= 3) * (
                sim.people.age < 12))  # cv.true() returns indices of people matching this condition, i.e. people under 20
        age_12_19 = cv.true((sim.people.age >= 12) * (sim.people.age < 20))
        age_20_29 = cv.true((sim.people.age >= 20) * (sim.people.age < 30))  # Multiplication means "and" here
        age_30_39 = cv.true((sim.people.age >= 30) * (sim.people.age < 40))  # Multiplication means "and" here
        age_40_49 = cv.true((sim.people.age >= 40) * (sim.people.age < 50))  # Multiplication means "and" here
        age_50_59 = cv.true((sim.people.age >= 50) * (sim.people.age < 60))  # Multiplication means "and" here
        age_60_69 = cv.true((sim.people.age >= 60) * (sim.people.age < 70))  # Multiplication means "and" here
        age_70_79 = cv.true((sim.people.age >= 70) * (sim.people.age < 80))  # Multiplication means "and" here
        age_80 = cv.true(sim.people.age >= 80)

        age_3_12 = age_3_12[:int(len(age_3_12) * vac_data[key_r][0])]
        age_12_19 = age_12_19[:int(len(age_12_19) * vac_data[key_r][1])]
        age_20_29 = age_20_29[:int(len(age_20_29) * vac_data[key_r][2])]
        age_30_39 = age_30_39[:int(len(age_30_39) * vac_data[key_r][3])]
        age_40_49 = age_40_49[:int(len(age_40_49) * vac_data[key_r][4])]
        age_50_59 = age_50_59[:int(len(age_50_59) * vac_data[key_r][5])]
        age_60_69 = age_60_69[:int(len(age_60_69) * vac_data[key_r][6])]
        age_70_79 = age_70_79[:int(len(age_70_79) * vac_data[key_r][7])]
        age_80 = age_80[:int(len(age_80) * vac_data[key_r][8])]
        inds = sim.people.uid  # Everyone in the population -- equivalent to np.arange(len(sim.people))
        vals = np.ones(len(sim.people))  # Create the array
        # update late by web scrambing more useful
        vals[age_3_12] = vac_data[key][0]  # 10% probability for people <50
        vals[age_12_19] = vac_data[key][1]
        vals[age_20_29] = vac_data[key][2]
        vals[age_30_39] = vac_data[key][3]
        vals[age_40_49] = vac_data[key][4]
        vals[age_50_59] = vac_data[key][5]
        vals[age_60_69] = vac_data[key][6]
        vals[age_70_79] = vac_data[key][7]
        vals[age_80] = vac_data[key][8]
        output = dict(inds=inds, vals=vals)
        # print('sinovac proportion is',vac_data[key_r][0],vac_data[key_r][1])
        return output

    def vaccinate_by_age_BioNTech(sim):
        key = nu + '_' + method + '_b'
        key_r = nu + '_' + method + '_r'
        age_3_12 = cv.true((sim.people.age >= 3) * (
                sim.people.age < 12))  # cv.true() returns indices of people matching this condition, i.e. people under 20
        age_12_19 = cv.true((sim.people.age >= 12) * (sim.people.age < 20))
        age_20_29 = cv.true((sim.people.age >= 20) * (sim.people.age < 30))  # Multiplication means "and" here
        age_30_39 = cv.true((sim.people.age >= 30) * (sim.people.age < 40))  # Multiplication means "and" here
        age_40_49 = cv.true((sim.people.age >= 40) * (sim.people.age < 50))  # Multiplication means "and" here
        age_50_59 = cv.true((sim.people.age >= 50) * (sim.people.age < 60))  # Multiplication means "and" here
        age_60_69 = cv.true((sim.people.age >= 60) * (sim.people.age < 70))  # Multiplication means "and" here
        age_70_79 = cv.true((sim.people.age >= 70) * (sim.people.age < 80))  # Multiplication means "and" here
        age_80 = cv.true(sim.people.age >= 80)

        age_3_12 = age_3_12[int(len(age_3_12) * vac_data[key_r][0]):]
        age_12_19 = age_12_19[int(len(age_12_19) * vac_data[key_r][1]):]
        age_20_29 = age_20_29[int(len(age_20_29) * vac_data[key_r][2]):]
        age_30_39 = age_30_39[int(len(age_30_39) * vac_data[key_r][3]):]
        age_40_49 = age_40_49[int(len(age_40_49) * vac_data[key_r][4]):]
        age_50_59 = age_50_59[int(len(age_50_59) * vac_data[key_r][5]):]
        age_60_69 = age_60_69[int(len(age_60_69) * vac_data[key_r][6]):]
        age_70_79 = age_70_79[int(len(age_70_79) * vac_data[key_r][7]):]
        age_80 = age_80[int(len(age_80) * vac_data[key_r][8]):]
        inds = sim.people.uid  # Everyone in the population -- equivalent to np.arange(len(sim.people))
        vals = np.ones(len(sim.people))  # Create the array
        # update late by web scrambing more useful
        vals[age_3_12] = vac_data[key][0]  # 10% probability for people <50
        vals[age_12_19] = vac_data[key][1]
        vals[age_20_29] = vac_data[key][2]
        vals[age_30_39] = vac_data[key][3]
        vals[age_40_49] = vac_data[key][4]
        vals[age_50_59] = vac_data[key][5]
        vals[age_60_69] = vac_data[key][6]
        vals[age_70_79] = vac_data[key][7]
        vals[age_80] = vac_data[key][8]
        output = dict(inds=inds, vals=vals)
        # print('biotect proportion is', vac_data[key_r][0], vac_data[key_r][1])
        return output

    def vaccinate_by_age_Sinovac_boost(sim):
        key = nu + '_' + method + '_sboost'
        age_3_12 = cv.true((sim.people.age >= 3) * (sim.people.age < 12) * (
                sim.people.doses == 2))  # cv.true() returns indices of people matching this condition, i.e. people under 20
        age_12_19 = cv.true((sim.people.age >= 12) * (sim.people.age < 20))
        age_20_29 = cv.true(
            (sim.people.age >= 20) * (sim.people.age < 30) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_30_39 = cv.true(
            (sim.people.age >= 30) * (sim.people.age < 40) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_40_49 = cv.true(
            (sim.people.age >= 40) * (sim.people.age < 50) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_50_59 = cv.true(
            (sim.people.age >= 50) * (sim.people.age < 60) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_60_69 = cv.true(
            (sim.people.age >= 60) * (sim.people.age < 70) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_70_79 = cv.true(
            (sim.people.age >= 70) * (sim.people.age < 80) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_80 = cv.true((sim.people.age >= 80) ** (sim.people.doses == 2))
        inds = sim.people.uid  # Everyone in the population -- equivalent to np.arange(len(sim.people))
        vals = np.ones(len(sim.people))  # Create the array
        # update late by web scrambing more useful
        vals[age_3_12] = vac_data[key][0]  # 10% probability for people <50
        vals[age_12_19] = vac_data[key][1]
        vals[age_20_29] = vac_data[key][2]
        vals[age_30_39] = vac_data[key][3]
        vals[age_40_49] = vac_data[key][4]
        vals[age_50_59] = vac_data[key][5]
        vals[age_60_69] = vac_data[key][6]
        vals[age_70_79] = vac_data[key][7]
        vals[age_80] = vac_data[key][8]
        output = dict(inds=inds, vals=vals)
        return output

    def vaccinate_by_age_BioNTech_boost(sim):
        key = nu + '_' + method + '_sboost'
        age_3_12 = cv.true((sim.people.age >= 3) * (sim.people.age < 12) * (
                sim.people.doses == 2))  # cv.true() returns indices of people matching this condition, i.e. people under 20
        age_12_19 = cv.true((sim.people.age >= 12) * (sim.people.age < 20) * (sim.people.doses == 2))
        age_20_29 = cv.true(
            (sim.people.age >= 20) * (sim.people.age < 30) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_30_39 = cv.true(
            (sim.people.age >= 30) * (sim.people.age < 40) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_40_49 = cv.true(
            (sim.people.age >= 40) * (sim.people.age < 50) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_50_59 = cv.true(
            (sim.people.age >= 50) * (sim.people.age < 60) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_60_69 = cv.true(
            (sim.people.age >= 60) * (sim.people.age < 70) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_70_79 = cv.true(
            (sim.people.age >= 70) * (sim.people.age < 80) * (sim.people.doses == 2))  # Multiplication means "and" here
        age_80 = cv.true((sim.people.age >= 80) * (sim.people.doses == 2))
        # distribute among the population with biotech veccines

        key = nu + '_' + method + '_sboost'
        key_s = nu + '_' + method + '_s'
        if third_vac_rate == '':
            dose_s_boost = int(vac_data[key][9])
        else:
            dose_s_boost = int(vac_data[key_s][9] / ful_vac_rate * third_vac_rate)
        dose_s = vac_data[key_s][9]
        # dose_s_second=vac_data[key_s][9]*vac_data[key_s][0]-dose_s_boost*vac_data[key][0]
        # dose_b_second=vac_data[nu+'_'+method+'_b'][9]*vac_data[nu+'_'+method+'_b'][0]
        # rate= dose_s_second/(dose_s_second+dose_b_second)

        age_3_12 = age_3_12[int(len(age_3_12) * (dose_s * vac_data[key_s][0] - dose_s_boost * vac_data[key][0]) / (
                    dose_s * vac_data[key_s][0] - dose_s_boost * vac_data[key][0] + vac_data[nu + '_' + method + '_b'][
                9] * vac_data[nu + '_' + method + '_b'][0])):]
        age_12_19 = age_12_19[int(len(age_12_19) * (dose_s * vac_data[key_s][1] - dose_s_boost * vac_data[key][1]) / (
                    dose_s * vac_data[key_s][1] - dose_s_boost * vac_data[key][1] + vac_data[nu + '_' + method + '_b'][
                9] * vac_data[nu + '_' + method + '_b'][1])):]
        age_20_29 = age_20_29[int(len(age_20_29) * (dose_s * vac_data[key_s][2] - dose_s_boost * vac_data[key][2]) / (
                    dose_s * vac_data[key_s][2] - dose_s_boost * vac_data[key][2] + vac_data[nu + '_' + method + '_b'][
                9] * vac_data[nu + '_' + method + '_b'][2])):]
        age_30_39 = age_30_39[int(len(age_30_39) * (dose_s * vac_data[key_s][3] - dose_s_boost * vac_data[key][3]) / (
                    dose_s * vac_data[key_s][3] - dose_s_boost * vac_data[key][3] + vac_data[nu + '_' + method + '_b'][
                9] * vac_data[nu + '_' + method + '_b'][3])):]
        age_40_49 = age_40_49[int(len(age_40_49) * (dose_s * vac_data[key_s][4] - dose_s_boost * vac_data[key][4]) / (
                    dose_s * vac_data[key_s][4] - dose_s_boost * vac_data[key][4] + vac_data[nu + '_' + method + '_b'][
                9] * vac_data[nu + '_' + method + '_b'][4])):]
        age_50_59 = age_50_59[int(len(age_50_59) * (dose_s * vac_data[key_s][5] - dose_s_boost * vac_data[key][5]) / (
                    dose_s * vac_data[key_s][5] - dose_s_boost * vac_data[key][5] + vac_data[nu + '_' + method + '_b'][
                9] * vac_data[nu + '_' + method + '_b'][5])):]
        age_60_69 = age_60_69[int(len(age_60_69) * (dose_s * vac_data[key_s][6] - dose_s_boost * vac_data[key][6]) / (
                    dose_s * vac_data[key_s][6] - dose_s_boost * vac_data[key][6] + vac_data[nu + '_' + method + '_b'][
                9] * vac_data[nu + '_' + method + '_b'][6])):]
        age_70_79 = age_70_79[int(len(age_70_79) * (dose_s * vac_data[key_s][7] - dose_s_boost * vac_data[key][7]) / (
                    dose_s * vac_data[key_s][7] - dose_s_boost * vac_data[key][7] + vac_data[nu + '_' + method + '_b'][
                9] * vac_data[nu + '_' + method + '_b'][7])):]
        age_80 = age_80[int(len(age_80) * (dose_s * vac_data[key_s][8] - dose_s_boost * vac_data[key][8]) / (
                    dose_s * vac_data[key_s][8] - dose_s_boost * vac_data[key][8] + vac_data[nu + '_' + method + '_b'][
                9] * vac_data[nu + '_' + method + '_b'][8])):]

        inds = sim.people.uid  # Everyone in the population -- equivalent to np.arange(len(sim.people))
        vals = np.ones(len(sim.people))  # Create the array
        vals[age_3_12] = vac_data[key][0]  # 10% probability for people <50
        vals[age_12_19] = vac_data[key][1]
        vals[age_20_29] = vac_data[key][2]
        vals[age_30_39] = vac_data[key][3]
        vals[age_40_49] = vac_data[key][4]
        vals[age_50_59] = vac_data[key][5]
        vals[age_60_69] = vac_data[key][6]
        vals[age_70_79] = vac_data[key][7]
        vals[age_80] = vac_data[key][8]
        output = dict(inds=inds, vals=vals)
        return output

    # close schoold
    # if soc_dis_rate==1:
    #     clo_sch_rate=0.5;clo_wo_rate=0.5;h_dis_rate=1
    # elif 1>soc_dis_rate>=0.3:
    #     clo_sch_rate=soc_dis_rate;clo_wo_rate=soc_dis_rate;h_dis_rate=soc_dis_rate
    # else:
    #     clo_sch_rate=0.05;clo_wo_rate=0.05;h_dis_rate=0.3
    clo_sch_rate, clo_wo_rate, h_dis_rate = soc_dis_rate, soc_dis_rate, soc_dis_rate
    if filename=='0.2':
        eff_days = ['2022-01-13', '2022-03-26','2022-04-05','2022-05-01','2022-08-01']
        eff_rates = [0.7, 0.1,0.7,0.85,soc_dis_rate]
    elif filename=='0.3':
        eff_days = ['2022-01-13', '2022-04-10','2022-04-20', '2022-05-01', '2022-08-01']
        eff_rates = [0.7, 0.1, 0.7, 0.85, soc_dis_rate]
    elif filename=='0.1':
        eff_days = ['2022-01-13',  '2022-05-01', '2022-08-01']
        eff_rates = [0.7,0.85, soc_dis_rate]
    close_schools = cv.change_beta(days=eff_days, changes=eff_rates, layers='s',
                                   do_plot=False)  # close 90% of schools
    close_works = cv.change_beta(days=eff_days, changes=eff_rates, layers='w',
                                 do_plot=False)  # close 80% of works

    social_distancing_home = cv.change_beta(days=eff_days, changes=eff_rates, layers='h')
    social_distancing_community = cv.change_beta(days=eff_days, changes=eff_rates,
                                                 layers='c')  # social distancing in communities 70% of community contact
 # social distancing in communities 70% of community contact

    # Define the vaccine
    # vaccine_S = cv.historical_vaccinate_prob(vaccine='sinovac',days=-30,subtarget=vaccinate_by_age_Sinovac,do_plot=False)
    # vaccine_B = cv.historical_vaccinate_prob(vaccine='biontech',days=-30,subtarget=vaccinate_by_age_BioNTech,do_plot=False)
    # dose_f = ful_vac_rate / 0.691956

    def num_doses1(sim):
        key = nu + '_' + method + '_s'
        if sim.t < 4:
            return int(vac_data[key][9] / 4)
        else:
            return 0

    def num_doses2(sim):
        key = nu + '_' + method + '_s'
        if 18 < sim.t < 21:
            return int(vac_data[key][9] / 2)
        else:
            return 0

    def num_doses2_boost(sim):
        key = nu + '_' + method + '_sboost'
        key_s = nu + '_' + method + '_s'
        if sim.t == 21:
            if third_vac_rate == '':
                return int(vac_data[key][9] / 1)
            else:
                print(int(vac_data[key_s][9] / ful_vac_rate) * third_vac_rate / 1)
                return int(vac_data[key_s][9] / ful_vac_rate * third_vac_rate / 1)
        else:
            return 0

    def num_doses3(sim):
        key = nu + '_' + method + '_b'
        if 21 < sim.t < 40:
            return int(vac_data[key][9] / 18)
        else:
            return 0

    def num_doses4(sim):
        key = nu + '_' + method + '_b'
        if 42 < sim.t < 61:
            return int(vac_data[key][9] / 18)
        else:
            return 0

    def num_doses4_boost(sim):
        key = nu + '_' + method + '_bboost'
        key_b = nu + '_' + method + '_b'
        if sim.t in [61, 62]:
            if third_vac_rate == '':
                return int(vac_data[key][9] / 2)
            else:
                print(int(vac_data[key_b][9] / ful_vac_rate) * third_vac_rate)
                return int(vac_data[key_b][9] / ful_vac_rate * third_vac_rate / 2)
        else:
            return 0

        # todo : vaccinenation shou be impremented

    vaccine_S_1 = cv.vaccinate_num(vaccine='sinovac', subtarget=vaccinate_by_age_Sinovac, num_doses=num_doses1,
                                   do_plot=False)
    vaccine_S_2 = cv.vaccinate_num(vaccine='sinovac', booster=True,
                                   subtarget={'inds': lambda sim: cv.true(sim.people.doses != 1), 'vals': 0},
                                   num_doses=num_doses2, do_plot=False)
    vaccine_S_2_boost = cv.vaccinate_num(vaccine='sinovac', booster=True, subtarget=vaccinate_by_age_Sinovac_boost,
                                         num_doses=num_doses2_boost, do_plot=False)
    vaccine_B_1 = cv.vaccinate_num(vaccine='biontech', subtarget=vaccinate_by_age_BioNTech, num_doses=num_doses3,
                                   do_plot=False)
    vaccine_B_2 = cv.vaccinate_num(vaccine='biontech', booster=True,
                                   subtarget={'inds': lambda sim: cv.true(sim.people.doses != 1), 'vals': 0},
                                   num_doses=num_doses4, do_plot=False)
    vaccine_B_2_boost = cv.vaccinate_num(vaccine='biontech', booster=True, subtarget=vaccinate_by_age_BioNTech_boost,
                                         num_doses=num_doses4_boost, do_plot=False)

    # Define the testing and contact tracing interventions  contact tracing is stopped
    tp_1 = cv.test_prob(symp_prob=0.3, asymp_prob=0.1, symp_quar_prob=0.5, asymp_quar_prob=0.1, test_delay=2,
                        do_plot=False)

    # test prob of symp_prob will test in two days,asymp_prob will test in seven days,wait for 3 days to know the result because conditional HK policy
    ct_1 = cv.contact_tracing(trace_probs=dict(h=0.8, s=0.3, w=0.3, c=0.2), presumptive=True, quar_period=3,
                              do_plot=False, start_day='2022-01-01', end_day='2022-02-16')
    ct_2 = cv.contact_tracing(trace_probs=dict(h=0.8, s=0.3, w=0.3, c=0.05), presumptive=True, quar_period=3,
                              do_plot=False, start_day='2022-02-16')
    # todo Universal Testing
    # tp_ut1 =cv.test_prob(symp_prob=0.3, asymp_prob=0.1, symp_quar_prob=0.5, asymp_quar_prob=0.1, test_delay=2,
    # do_plot=False,start_day='2022-01-01',end_day='2022-02-16')

    interventions = [vaccine_S_1, vaccine_S_2, vaccine_S_2_boost, vaccine_B_1, vaccine_B_2, vaccine_B_2_boost, tp_1,
                     ct_1, ct_2,
                     close_schools, close_works, social_distancing_community, social_distancing_home]
    sim.update_pars(interventions=interventions)
    sim.initialize()
    print('update sim successfully')
    return sim

    # peior vacination

    ##running with multisims

    row_num_age_new = 0

    # ,['new_infections','cum_infections','new_severe', 'n_severe','new_critical','n_critical','new_quarantined','n_quarantined','new_deaths','cum_deaths'],
    ##running with multisims
    # sim=cv.Sim(pars,label='Default')
    # msim=cv.MultiSim(sim)
    # msim.run(n_runs=5)
    # msim.mean()
    # for pic in ['new_infections','cum_infections','cum_vaccinated','new_severe', 'new_critical','new_deaths','new_vaccinated']:
    #     msim.plot_result(pic)

    # scenarios={'baseline':{'name':'Baseline','pars':{}},'high_vaccine':{'name':'vaccine_S_c','pars':{},} }
    # scens=cv.Scenarios(basepars=pars,scenarios=scenarios)
    # scens.run()
    # scens.plot(['new_infections','cum_infections','cum_vaccinated','new_severe', 'new_critical','new_deaths','new_vaccinated'])
    # fig=msim.plot(['new_infections','cum_infections','cum_vaccinated','new_severe', 'new_critical','new_deaths','new_vaccinated'])


day_stride = 21
pl.rcParams['font.size'] = 15
plotres = sc.objdict()
fig_path = 'fig1.png'
datafiles=sc.objdict()
plotres=sc.objdict()

def format_ax(ax, sim, key=None):
    ''' Format the axes nicely '''

    @ticker.FuncFormatter
    def date_formatter(x, pos):
        return (sim['start_day'] + dt.timedelta(days=x)).strftime('%b-%d')

    ax.xaxis.set_major_formatter(date_formatter)
    if key != 'r_eff':
        sc.commaticks()
    pl.xlim([17, 240])
    sc.boxoff()
    return


def plotter(key, sims, ax, ys=None, calib=False, label='', ylabel='', low_q=0.025, high_q=0.975, flabel=True,
            subsample=2):
    ''' Plot a single time series with uncertainty '''

    which = key.split('_')[1]

    if which == 'infections':
        color = '#e33d3e'
    elif which == 'severe':
        color = '#e6e600'
    elif which == 'critical':
        color= '#ff8000'
    elif which == 'deaths':
        color = '#1f1f1f'
    # if key == 'new_infections' or 'cum_infections':
    #     color= '#e33d3e'
    # elif key == 'new_severe' or 'n_severe' or 'cum_severe':
    #     color= '#e6e600'
    # elif key == 'new_critical' or 'n_critical' or 'cum_critical':
    #     color= '#ff8000'
    # elif key == 'new_deaths' or 'cum_deaths':
    #     color = '#1f1f1f'
    if ys is None:
        ys = []
        for i, s in enumerate(sims):
            if i < sims_cutoff:
                ys.append(s.results[key].values)
    #print('ys=', ys)
    yarr = np.array(ys)

    best = pl.median(yarr, axis=0)  # Changed from median to mean for smoother plots
    low = pl.quantile(yarr, q=low_q, axis=0)
    high = pl.quantile(yarr, q=high_q, axis=0)

    sim = sims[0]  # For having a sim to refer to

    tvec = np.arange(len(best))

    #data, data_t = None, None
    #if key in sim.data:
        #data_t = np.array((sim.data.index - sim['start_day']) / np.timedelta64(1, 'D'))
        #inds = np.arange(0, len(data_t), subsample)
        #data = sim.data[key][inds]
        #pl.plot(data_t[inds], data, 'd', c=color, markersize=10, alpha=0.5, label='Data')

    end = None
    if flabel:
        if which == 'infections':
            fill_label = '95% CI'
        else:
            fill_label = '95% CI'
    else:
        fill_label = None

    # Trim the beginning for r_eff and actually plot
    start = 2 if key == 'r_eff' else 0

    pl.fill_between(tvec[start:end], low[start:end], high[start:end], facecolor=color, alpha=0.2, label=fill_label)
    print('color is',color,'for', which)
    pl.plot(tvec[start:end], best[start:end], c=color, label=label, lw=4, alpha=1.0)

    sc.setylim()
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.arange(xmin, xmax, day_stride))
    pl.ylabel(ylabel)

    plotres[key] = sc.objdict(dict(tvec=tvec, best=best, low=low, high=high))

    return


def plot_intervs(sim, labels=True):
    ''' Plot interventions as vertical lines '''

    color = [0.1, 0.4, 0.1]
    mar12 = sim.day('2020-03-12')
    mar23 = sim.day('2020-03-23')
    #for day in [mar12, mar23]:
        #pl.axvline(day, c=color, linestyle='--', alpha=0.4, lw=3)

    if labels:
        yl = pl.ylim()
        labely = yl[1] * 1.03
        pl.text(mar12 - 25, labely, 'Schools close', color=color, alpha=0.9, style='italic')
        pl.text(mar23, labely, 'Stay-at-home', color=color, alpha=0.9, style='italic')
    return


# print_picture(factor=200,filename="test")

# for i in range(0,25):
# if i in np.arange(1):
# print_picture(n_beds=data[i][0],n_icus=data[i][1],ful_vac_rate=data[i][2],soc_dis_rate=data[i][3],factor=8,filename=str(i))

# para_source_data={0:[2000,255,0,0.7,'n'],1:[2000,255,0.692,0.7,'n'],2:[2000,255,0.692,1,'n'],3:[5000,500,0.692,0.7,'n'],4:[20000,2000,0.692,0.7,'n'],5:[2000,255,0.8,0.7,'sp'],6:[2000,255,0.8,0.7,'nm'],7:[2000,255,0.9,0.7,'sp'],8:[2000,255,0.9,0.7,'nm'],}
# para_data=pd.DataFrame(para_source_data)
#
# base case
df_list_age_cum = []
df_list_age_new = []
print_list_pars = []
# vaccination cate
if __name__ == '__main__':
    for i in test_para:
        if i in test_para:
            # ,6,7,8,9
            if para_data[i][3] == 'n':
                third_vac_rate = ''
            else:
                third_vac_rate = para_data[i][3]
            if para_data[i][5] == 'n':
                method = ''
            else:
                method = para_data[i][5]
            print(i, 'is successful in main')
            n_beds = para_data[i][0]
            n_icus = para_data[i][1]
            ful_vac_rate = para_data[i][2]
            third_vac_rate = third_vac_rate
            soc_dis_rate = para_data[i][4]
            factor = test_acu
            filename = str(i)
            method = method
            reps=7
            s0 = make_sim(n_beds, n_icus, ful_vac_rate, third_vac_rate, soc_dis_rate, factor, method=method)
            msim = cv.MultiSim(s0, n_runs=reps)
            msim.run(reseed=True, noise=0.0,  par_args={'ncpus': reps})
            sims = msim.sims
            #msim.plot(to_plot='overview', plot_sims=True)
            msim.reduce()
            sim = msim.base_sim


            print('sim run successfully')
            fig = pl.figure(num='HK pandemic situation and prediction', figsize=(30,20))
            tx1, ty1 = 0.01, 0.95
            tx2, ty2 = 0.50, 0.76
            ty3 = 0.57
            ty4 = 0.38
            ty5 = 0.19
            font_size = 30
            pl.figtext(tx1, ty1, 'a', fontsize=font_size)
            pl.figtext(tx2, ty1, 'b', fontsize=font_size)
            pl.figtext(tx1, ty2, 'c', fontsize=font_size)
            pl.figtext(tx2, ty2, 'd', fontsize=font_size)
            pl.figtext(tx1, ty3, 'e', fontsize=font_size)
            pl.figtext(tx2, ty3, 'f', fontsize=font_size)
            pl.figtext(tx1, ty4, 'g', fontsize=font_size)
            pl.figtext(tx2, ty4, 'h', fontsize=font_size)
            pl.figtext(tx1, ty5, 'i', fontsize=font_size)
            pl.figtext(tx2, ty5, 'j', fontsize=font_size)

            ##Fig a: daily infections
            x0, y0, dx, dy = 0.08, 0.81, 0.41, 0.16
            ax1 = pl.axes([x0, y0, dx, dy])
            format_ax(ax1, s0)
            plotter('new_infections', sims, ax1, calib=True, label='Model', ylabel='Daily infections')
            pl.legend(loc='upper right', frameon=False)

            # Fig b: cum infections:
            x02= 0.57
            ax2 = pl.axes([x02, y0, dx, dy])
            format_ax(ax2, s0)
            plotter('cum_infections', sims, ax2, calib=True, label='Model', ylabel='Cumulative infections')
            pl.legend(loc='lower right', frameon=False)



            ##fig C:Daily hospitalisations
            y02= 0.62
            ax3 = pl.axes([x0, y02, dx, dy])
            format_ax(ax3, sim)
            plotter('new_severe', sims, ax3, calib=True, label='Model', ylabel='Daily hospitalisations')
            pl.legend(loc='upper right', frameon=False)
            #pl.ylim([0, 130e3])
            #plot_intervs(sim)

            # fig D:
            ax4 = pl.axes([x02, y02, dx, dy])
            format_ax(ax4, sim)
            plotter('n_severe', sims, ax4, calib=True, label='Model', ylabel='Occupancy of hospitalisations')
            pl.legend(loc='upper right', frameon=False)
            # fig E:
            y03= 0.43
            ax5 = pl.axes([x0, y03, dx, dy])
            format_ax(ax5, sim)
            plotter('new_critical', sims, ax5, calib=True, label='Model', ylabel='Daily ICUs')
            pl.legend(loc='upper right', frameon=False)

            # fig F:

            ax6 = pl.axes([x02, y03, dx, dy])
            format_ax(ax6, sim)
            plotter('n_critical', sims, ax6, calib=True, label='Model', ylabel='Occupancy of ICUs')
            pl.legend(loc='upper right', frameon=False)

            # fig G:
            y04 = 0.24
            ax7 = pl.axes([x0, y04, dx, dy])
            format_ax(ax7, sim)
            plotter('cum_severe', sims, ax7, calib=True, label='Model', ylabel='Cumulative hospitalisations')
            pl.legend(loc='lower right', frameon=False)

            # fig H:
            ax8 = pl.axes([x02, y04, dx, dy])
            format_ax(ax8, sim)
            plotter('cum_critical', sims, ax8, calib=True, label='Model', ylabel='Cumulative ICUs')
            pl.legend(loc='lower right', frameon=False)

            # fig I:
            y05 = 0.05
            ax9 = pl.axes([x0, y05, dx, dy])
            format_ax(ax9, sim)
            plotter('new_deaths', sims, ax9, calib=True, label='Model', ylabel='Daily deaths')
            pl.legend(loc='upper right', frameon=False)

            # fig J:
            ax10 = pl.axes([x02, y05, dx, dy])
            format_ax(ax10, sim)
            plotter('cum_deaths', sims, ax10, calib=True, label='Model', ylabel='Cumulative deaths')
            pl.legend(loc='lower right', frameon=False)

            # %% Fig. 1b_B inserts (histograms)
            agehists = []
            for s, sim in enumerate(sims):
                agehist = sim['analyzers'][0]
                # if s == 0:
                # age_data = agehist.data
                agehists.append(agehist.hists[-1])

            # model outputs
            mposlist = []
            mdeathlist = []
            for hists in agehists:
                mposlist.append(hists['infectious'])
                mdeathlist.append(hists['dead'])
            mposarr = np.array(mposlist)
            mdeatharr = np.array(mdeathlist)

            low_q = 0.025
            high_q = 0.975
            mpbest = pl.median(mposarr, axis=0)
            mplow = pl.quantile(mposarr, q=low_q, axis=0)
            mphigh = pl.quantile(mposarr, q=high_q, axis=0)
            mdbest = pl.median(mdeatharr, axis=0)
            mdlow = pl.quantile(mdeatharr, q=low_q, axis=0)
            mdhigh = pl.quantile(mdeatharr, q=high_q, axis=0)

            w = 4
            off = 2

            # insets
            x0s, y0s, dxs, dys = 0.63, 0.88, 0.12, 0.10
            ax1s = pl.axes([x0s, y0s, dxs, dys])

            x = agehist.edges[:-1]
            xx = x + w - off
            pl.bar(xx - 2, mpbest, width=w, label='Model', facecolor='#e33d3e')
            for i, ix in enumerate(xx):
                pl.plot([ix-2, ix-2], [mplow[i], mphigh[i]], c='k')
            ax1s.set_xticks(np.arange(0, 81, 20))
            pl.xlabel('Age')
            pl.ylabel('Cases')
            sc.boxoff(ax1s)
            # pl.legend(frameon=False, bbox_to_anchor=(0.7, 1.1))

            y0sb = 0.12
            ax2s = pl.axes([x0s, y0sb, dxs, dys])
            pl.bar(xx - 2, mdbest, width=w, label='Model', facecolor='#666666')
            for i, ix in enumerate(xx):
                pl.plot([ix-2, ix-2], [mdlow[i], mdhigh[i]], c='k')
            ax2s.set_xticks(np.arange(0, 81, 20))
            pl.xlabel('Age')
            pl.ylabel('Deaths')
            sc.boxoff(ax2s)
            # pl.legend(frameon=False)
            sc.boxoff(ax1s)



            if do_save:
                cv.savefig(fig_path, dpi=300)

            if do_plot:
                pl.show()


            print_list_pars.append(s0.pars)
            file_name = filename + method + '.png'
            fig = s0.plot(start='2022-01-01', to_plot=to_plot, do_save=True, do_show=False, n_cols=2, figsize=(30, 20),
                          fig_path=file_name)

            # ,start = '2022-01-01',
            agehist = s0['analyzers'][0]

            row_num_age_cum = 0
            for date in age_trace_time:
                data = {'infected': agehist.get(date)['infectious'], 'severe': agehist.get(date)['severe'],
                        'critical': agehist.get(date)['critical'], 'dead': agehist.get(date)['dead']}
                df_list_age_cum.append([file_name, row_num_age_cum, pd.DataFrame(data)])
                row_num_age_cum += 12
            print('sim analyzer 1 successfully')
            agehist_new_df = s0['analyzers'][1].to_df()

            row_num_age_new = 0

            for date in age_trace_time:
                df_list_age_new.append([row_num_age_new, agehist_new_df[agehist_new_df['date'] == date]])
                row_num_age_new += 12
            print('sim analyzer 2 successfully')

# age-distribution data
# with pd.ExcelWriter('output_age.xlsx') as writer:
#     for filename, row_num_age_cum, df in df_list_age_cum:
#         df.to_excel(writer, sheet_name=filename[:-4], startrow=row_num_age_cum)

# cum-data new and accumulative
col_num_age_cum = 0
col_num_age_new = 0
cnt_1 = 0
with pd.ExcelWriter('output_age_cum.xlsx') as writer:
    for filename, row_num_age_cum, df in df_list_age_cum:
        df.to_excel(writer, startrow=row_num_age_cum, startcol=col_num_age_cum)
        cnt_1 += 1
        if cnt_1 % (len(age_trace_time)) == 0:  #
            col_num_age_cum += len(trace_state) + 3
print('print cum successfully')
cnt_2 = 0
with pd.ExcelWriter('output_age_new.xlsx') as writer:
    for row_num_age_new, df in df_list_age_new:
        df.to_excel(writer, startrow=row_num_age_new, startcol=col_num_age_new)
        cnt_2 += 1
        if cnt_2 % (len(age_trace_time)) == 0:  #
            col_num_age_new += len(trace_state) + 3
# todo finish
print('print new successfully')
print('done')
