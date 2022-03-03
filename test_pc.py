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

cv.check_version('3.1.2')

##input para:
do_plot = 1
do_save = 1
save_sim = 1
plot_hist = 0  # whether to keep people
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
age_trace_time = ['2022-02-28', '2022-03-07', '2022-03-15', '2022-03-23',
                  '2022-03-31', '2022-04-08', '2022-04-15', '2022-04-30',
                  '2022-05-15']  # trace the age-distribution data
sum_trace_time = ['2022-02-28', '2022-03-07', '2022-03-15', '2022-03-23',
                  '2022-03-31', '2022-04-08', '2022-04-15', '2022-04-30',
                  '2022-05-15']  # trace the new and accumulative cases
trace_state = ['infectious', 'severe', 'critical', 'dead']
vac_data = pd.read_excel(r'D:\onedrive\OneDrive - HKUST Connect\Desktop\IA\OneDrive - HKUST\PHD\ming\feature.xlsx',
                         sheet_name='vac_data')


# vac_data = pd.read_excel(r'feature.xlsx', sheet_name='vac_data')


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
        dur={'exp2inf': {'dist': 'lognormal_int', 'par1': 4.5, 'par2': 1.5},
             'inf2sym': {'dist': 'lognormal_int', 'par1': 1.1, 'par2': 0.9},
             'sym2sev': {'dist': 'lognormal_int', 'par1': 6.6, 'par2': 4.9},
             'sev2crit': {'dist': 'lognormal_int', 'par1': 1.5, 'par2': 2.0},
             'asym2rec': {'dist': 'lognormal_int', 'par1': 5.0, 'par2': 2.0},
             'mild2rec': {'dist': 'lognormal_int', 'par1': 5.0, 'par2': 2.0},
             'sev2rec': {'dist': 'lognormal_int', 'par1': 8.0, 'par2': 2.0},
             'crit2rec': {'dist': 'lognormal_int', 'par1': 15, 'par2': 6.3},
             'crit2die': {'dist': 'lognormal_int', 'par1': 10.7, 'par2': 4.8}},
        ##severity parameters
        n_beds_hosp=int(n_beds_hosp / factor),
        n_beds_icu=int(n_beds_icu / factor),
        no_hosp_factor=2,
        no_icu_factor=2,  # without beds critically cases will 2 times likely to die
    )

    sim = cv.Sim(pars=pars, location='china, hong kong special administrative region',
                 analyzers=[cv.age_histogram(days=age_trace_time, states=trace_state),
                            cv.daily_age_stats(states=trace_state)])  # use age-distribution data

    omicron1 = cv.variant('delta', days=sim.day('2022-01-15'), n_imports=16)  # np.array([16,15,125,98,97])
    omicron2 = cv.variant('delta', days=sim.day('2022-01-16'), n_imports=15)
    omicron3 = cv.variant('delta', days=sim.day('2022-01-17'), n_imports=125)
    omicron4 = cv.variant('delta', days=sim.day('2022-01-18'), n_imports=98)
    omicron5 = cv.variant('delta', days=sim.day('2022-01-19'), n_imports=97)
    omicron1.p['rel_beta'] = 7.018
    omicron1.p['rel_severe_prob'] = 0.2 * 2  ##1/7*0.47*3.2
    omicron1.p['rel_crit_prob'] = 0.36
    omicron1.p['rel_death_prob'] = 0.4 * 2
    omicron2.p['rel_beta'] = 7.018
    omicron2.p['rel_severe_prob'] = 0.2 * 2  ##1/7*0.47*3.2
    omicron2.p['rel_crit_prob'] = 0.36
    omicron2.p['rel_death_prob'] = 0.4 * 2
    omicron3.p['rel_beta'] = 7.018
    omicron3.p['rel_severe_prob'] = 0.2 * 2  ##1/7*0.47*3.2
    omicron3.p['rel_crit_prob'] = 0.36
    omicron3.p['rel_death_prob'] = 0.4 * 2
    omicron4.p['rel_beta'] = 7.018
    omicron4.p['rel_severe_prob'] = 0.2 * 2  ##1/7*0.47*3.2
    omicron4.p['rel_crit_prob'] = 0.36
    omicron4.p['rel_death_prob'] = 0.4 * 2
    omicron5.p['rel_beta'] = 7.018
    omicron5.p['rel_severe_prob'] = 0.2 * 2  ##1/7*0.47*3.2
    omicron5.p['rel_crit_prob'] = 0.36
    omicron5.p['rel_death_prob'] = 0.4 * 2
    sim['variants'] += [omicron1, omicron2, omicron3, omicron4, omicron5]

    if ful_vac_rate == 0.8:
        nu = '80%'
    elif ful_vac_rate == 0.9:
        nu = '90%'
    else:
        nu = 'base'

    def vaccinate_by_age_Sinovac(sim):
        key = nu + '_' + method + '_s'
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

        age_3_12 = age_3_12[:int(len(age_3_12) * 0.99)]
        age_12_19 = age_12_19[:int(len(age_12_19) * 0.11)]
        age_20_29 = age_20_29[:int(len(age_20_29) * 0.18)]
        age_30_39 = age_30_39[:int(len(age_30_39) * 0.26)]
        age_40_49 = age_40_49[:int(len(age_40_49) * 0.38)]
        age_50_59 = age_50_59[:int(len(age_50_59) * 0.48)]
        age_60_69 = age_60_69[:int(len(age_60_69) * 0.56)]
        age_70_79 = age_70_79[:int(len(age_70_79) * 0.61)]
        age_80 = age_80[:int(len(age_80) * 0.70)]
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

    def vaccinate_by_age_BioNTech(sim):
        key = nu + '_' + method + '_b'
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

        age_3_12 = age_3_12[int(len(age_3_12) * 0.99):]
        age_12_19 = age_12_19[int(len(age_12_19) * 0.11):]
        age_20_29 = age_20_29[int(len(age_20_29) * 0.18):]
        age_30_39 = age_30_39[int(len(age_30_39) * 0.26):]
        age_40_49 = age_40_49[int(len(age_40_49) * 0.38):]
        age_50_59 = age_50_59[int(len(age_50_59) * 0.48):]
        age_60_69 = age_60_69[int(len(age_60_69) * 0.56):]
        age_70_79 = age_70_79[int(len(age_70_79) * 0.61):]
        age_80 = age_80[int(len(age_80) * 0.70):]
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

    def vaccinate_by_age_Sinovac_c(sim):  # 7 days average rate
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
        inds = sim.people.uid  # Everyone in the population -- equivalent to np.arange(len(sim.people))
        vals = np.ones(len(sim.people))  # Create the array
        # update late by web scrambing more useful
        vals[age_3_12] = 0.0014  # 10% probability for people <50
        vals[age_12_19] = 0.0858
        vals[age_20_29] = 0.0388
        vals[age_30_39] = 0.0792
        vals[age_40_49] = 0.1094
        vals[age_50_59] = 0.1426
        vals[age_60_69] = 0.2561
        vals[age_70_79] = 0.4046
        vals[age_80] = 0.3971
        output = dict(inds=inds, vals=vals)
        return output

    def vaccinate_by_age_BioNTech_c(sim):
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
        inds = sim.people.uid  # Everyone in the population -- equivalent to np.arange(len(sim.people))
        vals = np.ones(len(sim.people))  # Create the array
        # update late by web scrambing more useful
        vals[age_3_12] = 0  # 10% probability for people <50
        vals[age_12_19] = 0.1173
        vals[age_20_29] = 0.1256
        vals[age_30_39] = 0.1404
        vals[age_40_49] = 0.101
        vals[age_50_59] = 0.0887
        vals[age_60_69] = 0.125
        vals[age_70_79] = 0.1353
        vals[age_80] = 0.0895
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
    close_schools = cv.change_beta(days=['2022-01-13', '2022-03-05'], changes=[0.7, clo_sch_rate], layers='s',
                                   do_plot=False)  # close 90% of schools
    close_works = cv.change_beta(days=['2022-01-13', '2022-03-05'], changes=[0.7, clo_wo_rate], layers='w',
                                 do_plot=False)  # close 80% of works

    social_distancing_home = cv.change_beta(days=['2022-01-13', '2022-03-05'], changes=[0.7, h_dis_rate], layers='h')
    social_distancing_community = cv.change_beta(days=['2022-01-13', '2022-03-05'], changes=[0.7, soc_dis_rate],
                                                 layers='c')  # social distancing in communities 70% of community contact

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
                return int(vac_data[key_s][9] / ful_vac_rate) * third_vac_rate / 1
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
                return int(vac_data[key_b][9] / ful_vac_rate) * third_vac_rate / 2
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


def print_picture(n_beds=2000, n_icus=255, ful_vac_rate=0.691956, soc_dis_rate=0.7, third_vac_rate='', factor=16,
                  filename='plot',
                  method=''):
    def protect_elderly(sim):
        if sim.t == sim.day('2022-04-01'):
            elderly = sim.people.age > 70
            sim.people.rel_sus[elderly] = 0.0

    """
    delta
    {'rel_beta': 2.2,
     'rel_symp_prob': 1.0,
     'rel_severe_prob': 3.2,
     'rel_crit_prob': 1.0,
     'rel_death_prob': 1.0}
    """
    # Create the simulation

    cv.options(jupyter=True, verbose=0)

    # peior vacination
    def get_boost(sim):
        if sim.people.doses == 1:
            sim.people.doses = 2

    ##running with multisims
    s0 = make_sim(n_beds, n_icus, ful_vac_rate, third_vac_rate, soc_dis_rate, factor, method=method)
    s0.run()
    print('sim run successfully')
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


# print_picture(factor=200,filename="test")

# for i in range(0,25):
# if i in np.arange(1):
# print_picture(n_beds=data[i][0],n_icus=data[i][1],ful_vac_rate=data[i][2],soc_dis_rate=data[i][3],factor=8,filename=str(i))
para_data = pd.read_excel(r'D:\onedrive\OneDrive - HKUST Connect\Desktop\IA\OneDrive - HKUST\PHD\ming\paras.xlsx',
                          header=1)
# para_data = pd.read_excel(r'paras.xlsx', header=1)
# para_source_data={0:[2000,255,0,0.7,'n'],1:[2000,255,0.692,0.7,'n'],2:[2000,255,0.692,1,'n'],3:[5000,500,0.692,0.7,'n'],4:[20000,2000,0.692,0.7,'n'],5:[2000,255,0.8,0.7,'sp'],6:[2000,255,0.8,0.7,'nm'],7:[2000,255,0.9,0.7,'sp'],8:[2000,255,0.9,0.7,'nm'],}
# para_data=pd.DataFrame(para_source_data)
#
# base case
df_list_age_cum = []
df_list_age_new = []
print_list_pars = []
# vaccination cate
for i in range(0, 25):
    if i in [1]:
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
        print_picture(n_beds=para_data[i][0], n_icus=para_data[i][1], ful_vac_rate=para_data[i][2],
                      third_vac_rate=third_vac_rate,
                      soc_dis_rate=para_data[i][4], factor=8, filename=str(i), method=method)
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
