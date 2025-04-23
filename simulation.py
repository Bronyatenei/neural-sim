
from netpyne import specs, sim
from netpyne import analysis
import numpy as np
import pandas as pd
import pickle
import gc
import matplotlib.pyplot as plt
import random

def create_model(scale=1):
    if not 1 <= scale <= 5:
        raise ValueError("Параметр scale должен быть от 1 до 5")
    netParams = specs.NetParams()
    # Настройка размера сети и цилиндрической области
    netParams.sizeX = 30000              #  x в мкм
    netParams.sizeY = 400000             # y в мкм
    netParams.sizeZ = 10000              # z в мкм
    netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 1, 'tau2': 10, 'e': 0}  # NMDA synaptic mechanism
    netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': 0.5, 'tau2': 5.0, 'e': -75}
    netParams.synMechParams['AMPA'] = {'mod': 'Exp2Syn', 'tau1': 0.8, 'tau2': 5.3, 'e': 0}
    # Объявляем глобальные переменные
    global numCells_inh_inter, numCells_exc_inter, numCells_alpha, numCells_gamma, numCells_stt, numCells_simp, numCells_sensory
    base_numCells_stt=50
    base_numCells_alpha=20
    base_numCells_gamma=10
    base_numCells_simp=15
    base_numCells_sensory = 75
    base_numCells_exc_inter = 50
    base_numCells_inh_inter = 50

    numCells_inh_inter = int(base_numCells_inh_inter * scale)
    numCells_exc_inter = int(base_numCells_exc_inter * scale)
    numCells_alpha = int(base_numCells_alpha * scale)
    numCells_gamma = int(base_numCells_gamma * scale)
    numCells_stt = int(base_numCells_stt * scale)
    numCells_simp = int(base_numCells_simp * scale)
    numCells_sensory = int(base_numCells_sensory * scale)

    # Данные клеток
    # Список параметров клеток: (cellType, diam, L, Ra, gnabar, gkbar, gl, el)
    cells_data = [
        ("C_fiber", 10, 20, 100.0, 0.12, 0.036, 0.0003, -65),
        ("spino_thalamic_neuron", 18, 18, 160, 0.1, 0.035, 0.00025, -70),
        ("alpha_motoneuron", 50, 60, 150, 0.18, 0.045, 0.0002, -70),
        ("gamma_motoneuron", 30, 40, 120, 0.15, 0.04, 0.00025, -65),
        ("simp", 25, 30, 200, 0.12, 0.08, 0.0003, -65),
        ("sensory_neuron", 20, 30, 100, 0.14, 0.04, 0.0003, -65),
        ("exc_inter_neuron", 20, 20, 100, 0.12, 0.036, 0.0003, -65),
        ("inh_inter_neuron", 15, 15, 120, 0.1, 0.03, 0.00025, -70)
    ]




    # Данные популяций
    # Список параметров популяций: (name, cellType, numCells, xRange_min, xRange_max, yRange_min, yRange_max, zRange_min, zRange_max)
    populations_data = [
        ("T_Interneuron_C1C8_l", "inh_inter_neuron", numCells_inh_inter, 13000, 15000, 0, 100000, 4000, 6000),
        ("T_Interneuron_C1C8_r", "inh_inter_neuron", numCells_inh_inter, 15000, 17000, 0, 100000, 4000, 6000),
        ("V_Interneuron_C1C8_l", "exc_inter_neuron", numCells_exc_inter, 13000, 15000, 0, 100000, 4000, 6000),
        ("V_Interneuron_C1C8_r", "exc_inter_neuron", numCells_exc_inter, 15000, 17000, 0, 100000, 4000, 6000),
        ("AlphaMN_C1_l", "alpha_motoneuron", numCells_alpha, 12000, 15000, 0, 10000, 2000, 5000),
        ("AlphaMN_C1_r", "alpha_motoneuron", numCells_alpha, 15000, 18000, 0, 10000, 2000, 5000),
        ("GammaMN_C1_l", "gamma_motoneuron", numCells_gamma, 12000, 15000, 0, 10000, 2000, 5000),
        ("GammaMN_C1_r", "gamma_motoneuron", numCells_gamma, 15000, 18000, 0, 10000, 2000, 5000),
        ("AlphaMN_C2C4_l", "alpha_motoneuron", numCells_alpha*3, 12000, 15000, 10000, 30000, 2000, 5000),
        ("AlphaMN_C2C4_r", "alpha_motoneuron", numCells_alpha*3, 15000, 18000, 10000, 30000, 2000, 5000),
        ("GammaMN_C2C4_l", "gamma_motoneuron", numCells_gamma*3, 12000, 15000, 10000, 30000, 2000, 5000),
        ("GammaMN_C2C4_r", "gamma_motoneuron", numCells_gamma*3, 15000, 18000, 10000, 30000, 2000, 5000),
        ("AlphaMN_C5C6_l", "alpha_motoneuron", numCells_alpha*2, 12000, 15000, 30000, 60000, 2000, 5000),
        ("AlphaMN_C5C6_r", "alpha_motoneuron", numCells_alpha*2, 15000, 18000, 30000, 60000, 2000, 5000),
        ("GammaMN_C5C6_l", "gamma_motoneuron", numCells_gamma*2, 12000, 15000, 30000, 60000, 2000, 5000),
        ("GammaMN_C5C6_r", "gamma_motoneuron", numCells_gamma*2, 15000, 18000, 30000, 60000, 2000, 5000),
        ("AlphaMN_C6C7_l", "alpha_motoneuron", numCells_alpha*2, 12000, 15000, 60000, 70000, 2000, 5000),
        ("AlphaMN_C6C7_r", "alpha_motoneuron", numCells_alpha*2, 15000, 18000, 60000, 70000, 2000, 5000),
        ("GammaMN_C6C7_l", "gamma_motoneuron", numCells_gamma*2, 12000, 15000, 60000, 70000, 2000, 5000),
        ("GammaMN_C6C7_r", "gamma_motoneuron", numCells_gamma*2, 15000, 18000, 60000, 70000, 2000, 5000),
        ("AlphaMN_C7C8_l", "alpha_motoneuron", numCells_alpha, 12000, 15000, 80000, 100000, 2000, 5000),
        ("AlphaMN_C7C8_r", "alpha_motoneuron", numCells_alpha, 15000, 18000, 80000, 100000, 2000, 5000),
        ("GammaMN_C7C8_l", "gamma_motoneuron", numCells_gamma, 12000, 15000, 80000, 100000, 2000, 5000),
        ("GammaMN_C7C8_r", "gamma_motoneuron", numCells_gamma, 15000, 18000, 80000, 100000, 2000, 5000),
        ("spino_thalamic_C1C8_l", "spino_thalamic_neuron", numCells_stt, 12500, 15000, 5000, 100000, 5000, 9000),
        ("spino_thalamic_C1C8_r", "spino_thalamic_neuron", numCells_stt, 15000, 17500, 5000, 100000, 5000, 9000),
        ("PSCT_C1C8_l", "spino_thalamic_neuron", numCells_stt, 12500, 15000, 5000, 100000, 5000, 9000),
        ("PSCT_C1C8_r", "spino_thalamic_neuron", numCells_stt, 15000, 17500, 5000, 100000, 5000, 9000),
        ("sensory_neuron_C1C8_l", "sensory_neuron", numCells_sensory, 0, 9000, 5000, 100000, 5000, 9000),
        ("sensory_neuron_C1C8_r", "sensory_neuron", numCells_sensory, 21000, 30000, 54000, 100000, 5000, 9000),
        ("R_AlphaMN_T1T12_l", "alpha_motoneuron", numCells_alpha*30, 12000, 15000, 100000, 300000, 2000, 5000),
        ("R_AlphaMN_T1T12_r", "alpha_motoneuron", numCells_alpha*30, 15000, 18000, 100000, 300000, 2000, 5000),
        ("R_GammaMN_T1T12_l", "gamma_motoneuron", numCells_gamma*30, 12000, 15000, 100000, 300000, 2000, 5000),
        ("R_GammaMN_T1T12_r", "gamma_motoneuron", numCells_gamma*30, 15000, 18000, 100000, 300000, 2000, 5000),
        ("S_AlphaMN_T1T6_l", "alpha_motoneuron", numCells_alpha*30, 12000, 15000, 100000, 200000, 2000, 5000),
        ("S_AlphaMN_T1T6_r", "alpha_motoneuron", numCells_alpha*30, 15000, 18000, 100000, 200000, 2000, 5000),
        ("S_GammaMN_T1T6_l", "gamma_motoneuron", numCells_gamma*30, 12000, 15000, 100000, 200000, 2000, 5000),
        ("S_GammaMN_T1T6_r", "gamma_motoneuron", numCells_gamma*30, 15000, 18000, 100000, 200000, 2000, 5000),
        ("S_AlphaMN_T7T12_l", "alpha_motoneuron", numCells_alpha*30, 12000, 15000, 200000, 300000, 2000, 5000),
        ("S_AlphaMN_T7T12_r", "alpha_motoneuron", numCells_alpha*30, 15000, 18000, 200000, 300000, 2000, 5000),
        ("S_GammaMN_T7T12_l", "gamma_motoneuron", numCells_gamma*30, 12000, 15000, 200000, 300000, 2000, 5000),
        ("S_GammaMN_T7T12_r", "gamma_motoneuron", numCells_gamma*30, 15000, 18000, 200000, 300000, 2000, 5000),
        ("simp_T1T6", "simp", numCells_simp*30, 12000, 18000, 100000, 200000, 2000, 5000),
        ("simp_T7T12", "simp", numCells_simp*30, 12000, 18000, 200000, 300000, 2000, 5000),
        ("spino_thalamic_T1T12_l", "spino_thalamic_neuron", numCells_stt*3, 12500, 15000, 100000, 300000, 5000, 9000),
        ("spino_thalamic_T1T12_r", "spino_thalamic_neuron", numCells_stt*3, 15000, 17500, 100000, 300000, 5000, 9000),
        ("PSCT_T1T12_l", "spino_thalamic_neuron", numCells_stt*3, 12500, 15000, 100000, 300000, 5000, 9000),
        ("PSCT_T1T12_r", "spino_thalamic_neuron", numCells_stt*3, 15000, 17500, 100000, 300000, 5000, 9000),
        ("sensory_neuron_T1T12_l", "sensory_neuron", numCells_sensory*2, 0, 9000, 100000, 300000, 5000, 9000),
        ("sensory_neuron_T1T12_r", "sensory_neuron", numCells_sensory*2, 21000, 30000, 100000, 300000, 5000, 9000),
        ("T_Interneuron_T1T12_l", "inh_inter_neuron", numCells_inh_inter, 13000, 15000, 100000, 300000, 4000, 6000),
        ("T_Interneuron_T1T12_r", "inh_inter_neuron", numCells_inh_inter, 15000, 17000, 100000, 300000, 4000, 6000),
        ("V_Interneuron_T1T12_l", "exc_inter_neuron", numCells_exc_inter, 13000, 15000, 100000, 300000, 4000, 6000),
        ("V_Interneuron_T1T12_r", "exc_inter_neuron", numCells_exc_inter, 15000, 17000, 100000, 300000, 4000, 6000),
        ("N_AlphaMN_L1L3_l", "alpha_motoneuron", numCells_alpha*20, 12000, 15000, 300000, 330000, 2000, 5000),
        ("N_AlphaMN_L1L3_r", "alpha_motoneuron", numCells_alpha*20, 15000, 18000, 300000, 330000, 2000, 5000),
        ("N_GammaMN_L1L3_l", "gamma_motoneuron", numCells_gamma*20, 12000, 15000, 300000, 330000, 2000, 5000),
        ("N_GammaMN_L1L3_r", "gamma_motoneuron", numCells_gamma*20, 15000, 18000, 300000, 330000, 2000, 5000),
        ("N_AlphaMN_L3L4_l", "alpha_motoneuron", numCells_alpha*10, 12000, 15000, 330000, 340000, 2000, 5000),
        ("N_AlphaMN_L3L4_r", "alpha_motoneuron", numCells_alpha*10, 15000, 18000, 330000, 340000, 2000, 5000),
        ("N_GammaMN_L3L4_l", "gamma_motoneuron", numCells_gamma*10, 12000, 15000, 330000, 340000, 2000, 5000),
        ("N_GammaMN_L3L4_r", "gamma_motoneuron", numCells_gamma*10, 15000, 18000, 330000, 340000, 2000, 5000),
        ("N_AlphaMN_L4L6_l", "alpha_motoneuron", numCells_alpha*20, 12000, 15000, 340000, 360000, 2000, 5000),
        ("N_AlphaMN_L4L6_r", "alpha_motoneuron", numCells_alpha*20, 15000, 18000, 340000, 360000, 2000, 5000),
        ("N_GammaMN_L4L6_l", "gamma_motoneuron", numCells_gamma*20, 12000, 15000, 340000, 360000, 2000, 5000),
        ("N_GammaMN_L4L6_r", "gamma_motoneuron", numCells_gamma*20, 15000, 18000, 340000, 360000, 2000, 5000),
        ("spino_thalamic_L1L6_l", "spino_thalamic_neuron", numCells_stt*3, 12500, 15000, 300000, 360000, 5000, 9000),
        ("spino_thalamic_L1L6_r", "spino_thalamic_neuron", numCells_stt*3, 15000, 17500, 300000, 360000, 5000, 9000),
        ("PSCT_L1L6_l", "spino_thalamic_neuron", numCells_stt*3, 12500, 15000, 300000, 360000, 5000, 9000),
        ("PSCT_L1L6_r", "spino_thalamic_neuron", numCells_stt*3, 15000, 17500, 300000, 360000, 5000, 9000),
        ("sensory_neuron_L1L6_l", "sensory_neuron", numCells_sensory*2, 0, 9000, 300000, 360000, 5000, 9000),
        ("sensory_neuron_L1L6_r", "sensory_neuron", numCells_sensory*2, 21000, 30000, 300000, 360000, 5000, 9000),
        ("T_Interneuron_L1L6_l", "inh_inter_neuron", numCells_inh_inter, 13000, 15000, 300000, 360000, 4000, 6000),
        ("T_Interneuron_L1L6_r", "inh_inter_neuron", numCells_inh_inter, 15000, 17000, 300000, 360000, 4000, 6000),
        ("V_Interneuron_L1L6_l", "exc_inter_neuron", numCells_exc_inter, 13000, 15000, 300000, 360000, 4000, 6000),
        ("V_Interneuron_L1L6_r", "exc_inter_neuron", numCells_exc_inter, 15000, 17000, 300000, 360000, 4000, 6000)


    ]

    # Данные связей
    connections_data = [
        ("sensory_to_PSCT_C1C8_l", "sensory_neuron_C1C8_l", "PSCT_C1C8_l_p", 0.8, 0.2, "dist_3D/(30 * 1000)", "AMPA"),
        ("sensory_to_PSCT_C1C8_r", "sensory_neuron_C1C8_r", "PSCT_C1C8_r", 0.8, 0.2, "dist_3D/(30 * 1000)", "AMPA"),
        ("sensory_neuron_to_exc_C1C8_r", "sensory_neuron_C1C8_r", "V_Interneuron_C1C8_r", 0.6, 0.2, "dist_3D/(5 * 1000)", "exc"),
        ("sensory_neuron_to_exc_C1C8_l", "sensory_neuron_C1C8_l", "V_Interneuron_C1C8_l", 0.6, 0.2, "dist_3D/(5 * 1000)", "exc"),
        ("exc_to_stt_C1C8_r", "V_Interneuron_C1C8_r", "spino_thalamic_C1C8_l", 0.7, 0.1, "dist_3D/(5 * 1000)", "AMPA"),
        ("exc_to_stt_C1C8_l", "V_Interneuron_C1C8_l", "spino_thalamic_C1C8_r", 0.7, 0.1, "dist_3D/(5 * 1000)", "AMPA"),
        ("sensory_to_PSCT_T1T12_l", "sensory_neuron_T1T12_l", "PSCT_T1T12_l_p", 0.8, 0.2, "dist_3D/(30 * 1000)", "AMPA"),
        ("sensory_to_PSCT_T1T12_r", "sensory_neuron_T1T12_r", "PSCT_T1T12_r", 0.8, 0.2, "dist_3D/(30 * 1000)", "AMPA"),
        ("sensory_neuron_to_exc_T1T12_r", "sensory_neuron_T1T12_r", "V_Interneuron_T1T12_r", 0.6, 0.2, "dist_3D/(5 * 1000)", "exc"),
        ("sensory_neuron_to_exc_T1T12_l", "sensory_neuron_T1T12_l",  "V_Interneuron_T1T12_l", 0.6, 0.2, "dist_3D/(5 * 1000)", "exc"),
        ("exc_to_stt_T1T12_r", "V_Interneuron_T1T12_r", "spino_thalamic_T1T12_l", 0.7, 0.1, "dist_3D/(5 * 1000)", "AMPA"),
        ("exc_to_stt_T1T12_l", "V_Interneuron_T1T12_l", "spino_thalamic_T1T12_r", 0.7, 0.1, "dist_3D/(5 * 1000)", "AMPA"),
        ("sensory_to_PSCT_L1L6_l", "sensory_neuron_L1L6_l", "PSCT_L1L6_l_p", 0.8, 0.2, "dist_3D/(30 * 1000)", "AMPA"),
        ("sensory_to_PSCT_L1L6_r", "sensory_neuron_L1L6_r", "PSCT_L1L6_r", 0.8, 0.2, "dist_3D/(30 * 1000)", "AMPA"),
        ("sensory_neuron_to_exc_L1L6_r", "sensory_neuron_L1L6_r", "V_Interneuron_L1L6_r", 0.6, 0.2, "dist_3D/(5 * 1000)", "exc"),
        ("sensory_neuron_to_exc_L1L6_l", "sensory_neuron_L1L6_l", "V_Interneuron_L1L6_l", 0.6, 0.2, "dist_3D/(5 * 1000)", "exc"),
        ("exc_to_stt_L1L6_r", "V_Interneuron_L1L6_r", "spino_thalamic_L1L6_l", 0.7, 0.1, "dist_3D/(5 * 1000)", "AMPA"),
        ("exc_to_stt_L1L6_l", "V_Interneuron_L1L6_l", "spino_thalamic_L1L6_r", 0.7, 0.1, "dist_3D/(5 * 1000)", "AMPA"),
        ("inh_to_stt_L1L6_r", "T_Interneuron_L1L6_r", "spino_thalamic_L1L6_l", 0.5, 0.08, "dist_3D/(5 * 1000)", "inh"),
        ("inh_to_stt_L1L6_l", "T_Interneuron_L1L6_l", "spino_thalamic_L1L6_r", 0.5, 0.08, "dist_3D/(5 * 1000)", "inh")

    ]


    # Создаём клетки
    for cell in cells_data:
        cellType, diam, L, Ra, gnabar, gkbar, gl, el = cell
        secs = {}
        secs['soma'] = {'geom': {}, 'mechs': {}}
        secs['soma']['geom'] = {'diam': diam, 'L': L, 'Ra': Ra}
        secs['soma']['mechs']['hh'] = {'gnabar': gnabar, 'gkbar': gkbar, 'gl': gl, 'el': el}
        netParams.cellParams[cellType] = {'secs': secs}
    print("Все параметры клеток созданы.")

    # Создаём популяции
    for pop in populations_data:
        name, cellType, numCells, x_min, x_max, y_min, y_max, z_min, z_max = pop
        netParams.popParams[name] = {
            'cellType': cellType,
            'numCells': numCells,
            'xRange': [x_min, x_max],
            'yRange': [y_min, y_max],
            'zRange': [z_min, z_max]
        }
    print("Все популяции созданы.")

    # Создаём связи
    for conn in connections_data:
        name, prePop, postPop, probability, weight, delay, synMech = conn
        netParams.connParams[name] = {
            'preConds': {'pop': prePop},
            'postConds': {'pop': postPop},
            'probability': probability,
            'weight': weight,
            'delay': delay,
            'synMech': synMech
        }
    print("Все связи созданы.")

    print(f"Основная модель создана с масштабом {scale}.")
    return netParams

def set_background_activity(netParams, activity_scale=1):

    # Проверяем, что activity_scale в допустимом диапазоне
    if not 1 <= activity_scale <= 3:
        raise ValueError("Параметр activity_scale должен быть от 1 до 3")

        # Данные стимуляций Источники
    stim_sources_data = [
        ("to_sensop_netstim_1", "NetStim", 40, 0.8),
        ("to_alpha1_netstim_1", "NetStim", 15, 0.2),
        ("to_alpha1_netstim_2", "NetStim", 15, 0.2),
        ("to_alpha1_netstim_3", "NetStim", 20, 0.5),
        ("to_alpha1_netstim_4", "NetStim", 10, 0.6),
        ("to_alpha1_netstim_5", "NetStim", 10, 0.6),
        ("to_alpha2_netstim", "NetStim", 15, 0.1),
        ("netstim_back_1", "NetStim", 10, 0.4),
        ("netstim_back_2", "NetStim", 10, 0.4),
        ("simp_netstim_1", "NetStim", 10, 0.1),
        ("simp_netstim_2", "NetStim", 7, 0.4),
        ("sensor_netstim_2", "NetStim", 40, 0.8),
        ("netstim_leg_1", "NetStim", 20, 0.4),
        ("netstim_leg_2", "NetStim", 20, 0.4),
        ("netstim_leg_3", "NetStim", 20, 0.4),
        ("sensor_netstim_3", "NetStim", 40, 0.8)

    ]
    # Цели
    stim_targets_data = [
        ("to_sensop_netstim_1_r", "to_sensop_netstim_1", "sensory_neuron_C1C8_r", 0.2),
        ("to_sensop_netstim_1_l", "to_sensop_netstim_1", "sensory_neuron_C1C8_l", 0.2),
        ("to_alpha1_netstim_1a_l", "to_alpha1_netstim_1", "AlphaMN_C1_l", 0.3),
        ("to_alpha1_netstim_1a_r", "to_alpha1_netstim_1", "AlphaMN_C1_r", 0.3),
        ("to_alpha1_netstim_1g_l", "to_alpha1_netstim_1", "GammaMN_C1_l", 0.3),
        ("to_alpha1_netstim_1g_r", "to_alpha1_netstim_1", "GammaMN_C1_r", 0.3),
        ("to_alpha1_netstim_2a_l", "to_alpha1_netstim_2", "AlphaMN_C2C4_l", 0.2),
        ("to_alpha1_netstim_2a_r", "to_alpha1_netstim_2", "AlphaMN_C2C4_r", 0.2),
        ("to_alpha1_netstim_2g_l", "to_alpha1_netstim_2", "GammaMN_C2C4_l", 0.2),
        ("to_alpha1_netstim_2g_r", "to_alpha1_netstim_2", "GammaMN_C2C4_r", 0.2),
        ("to_alpha1_netstim_3a_l", "to_alpha1_netstim_3", "AlphaMN_C5C6_l", 0.3),
        ("to_alpha1_netstim_3a_r", "to_alpha1_netstim_3", "AlphaMN_C5C6_r", 0.3),
        ("to_alpha1_netstim_3g_l", "to_alpha1_netstim_3", "GammaMN_C5C6_l", 0.3),
        ("to_alpha1_netstim_3g_r", "to_alpha1_netstim_3", "GammaMN_C5C6_r", 0.3),
        ("to_alpha1_netstim_4a_l", "to_alpha1_netstim_4", "AlphaMN_C6C7_l", 0.3),
        ("to_alpha1_netstim_4a_r", "to_alpha1_netstim_4", "AlphaMN_C6C7_r", 0.3),
        ("to_alpha1_netstim_4g_l", "to_alpha1_netstim_4", "GammaMN_C6C7_l", 0.3),
        ("to_alpha1_netstim_4g_r", "to_alpha1_netstim_4", "GammaMN_C6C7_r", 0.3),
        ("to_alpha1_netstim_5a_l", "to_alpha1_netstim_5", "AlphaMN_C7C8_l", 0.3),
        ("to_alpha1_netstim_5a_r", "to_alpha1_netstim_5", "AlphaMN_C7C8_r", 0.3),
        ("to_alpha1_netstim_5g_l", "to_alpha1_netstim_5", "GammaMN_C7C8_l", 0.3),
        ("to_alpha1_netstim_5g_r", "to_alpha1_netstim_5", "GammaMN_C7C8_r", 0.3),
        ("netstim_leg_1a_l", "netstim_leg_1", "N_AlphaMN_L1L3_l", 0.3),
        ("netstim_leg_1a_r", "netstim_leg_1", "N_AlphaMN_L1L3_r", 0.3),
        ("netstim_leg_1g_l", "netstim_leg_1", "N_GammaMN_L1L3_l", 0.3),
        ("netstim_leg_1g_r", "netstim_leg_1", "N_GammaMN_L1L3_r", 0.3),
        ("netstim_leg_2a_l", "netstim_leg_2", "N_AlphaMN_L3L4_l", 0.3),
        ("netstim_leg_2a_r", "netstim_leg_2", "N_AlphaMN_L3L4_r", 0.3),
        ("netstim_leg_2g_l", "netstim_leg_2", "N_GammaMN_L3L4_l", 0.3),
        ("netstim_leg_2g_r", "netstim_leg_2", "N_GammaMN_L3L4_r", 0.3),
        ("netstim_leg_3a_l", "netstim_leg_3", "N_AlphaMN_L4L6_l", 0.3),
        ("netstim_leg_3a_r", "netstim_leg_3", "N_AlphaMN_L4L6_r", 0.3),
        ("netstim_leg_3g_l", "netstim_leg_3", "N_GammaMN_L4L6_l", 0.3),
        ("netstim_leg_3g_r", "netstim_leg_3", "N_GammaMN_L4L6_r", 0.3),
        ("sensor_netstim_3_r", "sensor_netstim_3", "sensory_neuron_L1L6_r", 0.2),
        ("sensor_netstim_3_l", "sensor_netstim_3", "sensory_neuron_L1L6_l", 0.2)


    ]

    # Определяем максимальные значения rate для каждой группы
    max_rates = {
        "to_sensop_netstim_1": 100,  # Сенсорные нейроны
        "sensor_netstim_2": 100,
        "sensor_netstim_3": 100,
        "to_alpha1_netstim_1": 60,  # Альфа- и гамма-мотонейроны
        "to_alpha1_netstim_2": 60,
        "to_alpha1_netstim_3": 80,
        "to_alpha1_netstim_4": 40,
        "to_alpha1_netstim_5": 40,
        "to_alpha2_netstim": 60,
        "netstim_back_1": 40,  # Спинальные нейроны (спина)
        "netstim_back_2": 40,
        "netstim_leg_1": 60,  # Ноги
        "netstim_leg_2": 80,
        "netstim_leg_3": 60,
        "simp_netstim_1": 40,  # Симпатические нейроны
        "simp_netstim_2": 30,
    }

    # Создаём источники стимуляции с масштабированием
    for source in stim_sources_data:
        name, stim_type, base_rate, base_noise = source

        # Масштабируем rate
        max_rate = max_rates[name]
        scaled_rate = base_rate + (max_rate - base_rate) * (activity_scale - 1) / 2
        scaled_rate = min(scaled_rate, max_rate)  # Ограничиваем сверху

        # Масштабируем noise (увеличиваем на 0.1 за каждый уровень, но не больше 1)
        scaled_noise = min(base_noise + 0.1 * (activity_scale - 1), 1.0)

        netParams.stimSourceParams[name] = {
            'type': stim_type,
            'rate': scaled_rate,
            'noise': scaled_noise
        }
    print(f"Источники фоновой активности созданы с activity_scale={activity_scale}.")

    # Создаём цели стимуляции
    for target in stim_targets_data:
        name, source_name, pop, weight = target
        netParams.stimTargetParams[name] = {
            'source': source_name,
            'conds': {'pop': pop},
            'weight': weight,
            'delay': "uniform(0, 10)",
            'sec': 'soma',
            'loc': 0.5
        }
    print("Все цели стимуляции созданы.")
    return netParams

def set_pain(netParams, vertebrae, pain_intensity=1, side="both", pain_duration=500, sim_duration=1000):
    
    #Добавляет болевой сигнал через C-волокна в указанных позвонках.

    #Args:
    #    netParams: Параметры сети
    #    vertebrae (list): Список позвонков, где генерируется боль (например, ["C1", "C2"])
    #    pain_intensity (float): Интенсивность боли (от 1 до 3)
    #    side (str): Сторона ("left", "right", "both"), по умолчанию "both"
    
    vertebrae_y_ranges = {
    "C1": [0, 12000],
    "C2": [12000, 24000],
    "C3": [24000, 36000],
    "C4": [36000, 48000],
    "C5": [48000, 60000],
    "C6": [60000, 72000],
    "C7": [72000, 84000],
    "C8": [84000, 96000],
    "T1": [96000, 111000],
    "T2": [111000, 126000],
    "T3": [126000, 141000],
    "T4": [141000, 156000],
    "T5": [156000, 171000],
    "T6": [171000, 186000],
    "T7": [186000, 201000],
    "T8": [201000, 216000],
    "T9": [216000, 231000],
    "T10": [231000, 246000],
    "T11": [246000, 261000],
    "T12": [261000, 276000],
    "L1": [276000, 290000],
    "L2": [290000, 304000],
    "L3": [304000, 318000],
    "L4": [318000, 332000],
    "L5": [332000, 346000],
    "L6": [346000, 360000]
    }
    if not 1 <= pain_intensity <= 3:
        raise ValueError("Параметр pain_intensity должен быть от 1 до 3")

    if side not in ["left", "right", "both"]:
        raise ValueError("Параметр side должен быть 'left', 'right' или 'both'")

    # Проверяем, что указанные позвонки существуют
    for vertebra in vertebrae:
        if vertebra not in vertebrae_y_ranges:
            raise ValueError(f"Позвонок {vertebra} не найден в vertebrae_y_ranges")

    # Параметры C-волокон
    secs_C = {}
    secs_C['soma'] = {'geom': {}, 'mechs': {}}
    secs_C['soma']['geom'] = {'diam': 10, 'L': 20, 'Ra': 100.0}
    secs_C['soma']['mechs']['hh'] = {
        'gnabar': 0.12,
        'gkbar': 0.036,
        'gl': 0.0003,
        'el': -65
    }
    netParams.cellParams['C_fiber'] = {'secs': secs_C}

    # Определяем отделы
    sections = {
        "C1C8": [v for v in vertebrae_y_ranges if v.startswith("C")],
        "T1T12": [v for v in vertebrae_y_ranges if v.startswith("T")],
        "L1L6": [v for v in vertebrae_y_ranges if v.startswith("L")]
    }
    # Инициализируем pain_info
    pain_info = {}
    print("Инициализирован pain_info:", pain_info)
    # Создаём популяции C-волокон и стимуляции
    print("Создаём популяции и стимуляции...")
    for vertebra in vertebrae:
        y_min, y_max = vertebrae_y_ranges[vertebra]
        sides = ["left", "right"] if side == "both" else [side]

        # Определяем отдел для текущего позвонка
        if vertebra.startswith("C"):
            section = "C1C8"
        elif vertebra.startswith("T"):
            section = "T1T12"
        else:  # L
            section = "L1L6"
        print(f"  Отдел: {section}")

        for s in sides:
            # Координаты C-волокон (в ганглиях)
            x_range = [0, 9000] if s == "left" else [21000, 30000]
            pop_name = f"C_fiber_{vertebra}_{s[0]}"  # Например, "C_fiber_C1_l"

            # Создаём популяцию C-волокон
            netParams.popParams[pop_name] = {
                'cellType': "C_fiber",
                'numCells': 75,
                'xRange': x_range,
                'yRange': [y_min, y_max],
                'zRange': [2000, 5000]
            }

            # Добавляем стимуляцию через NetStim
            stim_name = f"pain_stim_{pop_name}"
            base_rate = 30
            max_rate = 150
            scaled_rate = base_rate + (max_rate - base_rate) * (pain_intensity - 1) / 2
            scaled_rate = min(scaled_rate, max_rate)
            # Время начала стимуляции: случайное значение от 0 до (sim_duration/2 - pain_duration)
            max_start = max(200, sim_duration / 2 - pain_duration / 2)
            start_time = random.uniform(100, max_start) if max_start > 0 else 0
            print(f" start_time = {start_time}")
            # Пересчитываем длительность в количество спайков
            num_spikes = scaled_rate * (pain_duration / 1000)  # rate в Гц, duration в мс
            print(f"    stim_name: {stim_name}, scaled_rate: {scaled_rate}, num_spikes: {num_spikes}, start_time: {start_time}")

            # Сохраняем в pain_info
            pain_info[stim_name] = {
                'start': start_time,
                'duration': pain_duration
            }
            print(f"    Добавлено в pain_info: {stim_name} -> {pain_info[stim_name]}")

            netParams.stimSourceParams[stim_name] = {
                'type': 'NetStim',
                'rate': scaled_rate,
                'noise': 0.7,
                'start': start_time,
                'number': num_spikes
            }
            netParams.stimTargetParams[f"{stim_name}_to_{pop_name}"] = {
                'source': stim_name,
                'conds': {'pop': pop_name},
                'weight': 0.2,
                'delay': 'uniform(0, 30)',
                'sec': 'soma',
                'loc': 0.5
            }

            # Создаём связи с другими популяциями
            # C-волокна → Спинно-таламические нейроны (перекрёстное соединение: справа → слева, слева → справа)
            stn_pop = f"spino_thalamic_{section}_l" if s == "right" else f"spino_thalamic_{section}_r"
            if stn_pop in netParams.popParams:
                # Ограничиваем соединения по расстоянию (в пределах позвонка)
                vertebra_length = y_max - y_min  # Длина позвонка (например, 12000 мкм для шейных)
                netParams.connParams[f"C_to_STN_{vertebra}_{s[0]}"] = {
                    'preConds': {'pop': pop_name},
                    'postConds': {'pop': stn_pop},
                    'probability': f"1.0 if dist_3D < {vertebra_length} else 0.0",
                    'weight': 0.3,
                    'delay': "dist_3D/(30 * 1000)",
                    'synMech': "AMPA",
                    'sec': 'soma',
                    'loc': 0.5
                }

            # C-волокна → Возбуждающие интернейроны (на той же стороне)
            v_inter_pop = f"V_Interneuron_{section}_{s[0]}"
            if v_inter_pop in netParams.popParams:
                netParams.connParams[f"C_to_V_Inter_{vertebra}_{s[0]}"] = {
                    'preConds': {'pop': pop_name},
                    'postConds': {'pop': v_inter_pop},
                    'probability': f"1.0 if dist_3D < {vertebra_length} else 0.0",
                    'weight': 0.2,
                    'delay': "dist_3D/(30 * 1000)",
                    'synMech': "AMPA",
                    'sec': 'soma',
                    'loc': 0.5
                }

            # C-волокна → Тормозные интернейроны (на той же стороне)
            t_inter_pop = f"T_Interneuron_{section}_{s[0]}"
            if t_inter_pop in netParams.popParams:
                netParams.connParams[f"C_to_T_Inter_{vertebra}_{s[0]}"] = {
                    'preConds': {'pop': pop_name},
                    'postConds': {'pop': t_inter_pop},
                    'probability': f"1.0 if dist_3D < {vertebra_length} else 0.0",
                    'weight': 0.2,
                    'delay': "dist_3D/(30 * 1000)",
                    'synMech': "AMPA",
                    'sec': 'soma',
                    'loc': 0.5
                }
    print("Итоговый pain_info в set_pain:", pain_info)
    return netParams, pain_info

def set_electrode(x=15000, y=50000, z=3500, num_electrodes=1, spacing=5000, axis="y"):
    """
    Задаёт координаты виртуального электрода или цепочки электродов.

    Args:
        x (float): Координата X начальной точки (по умолчанию 15000, центр между левой и правой сторонами)
        y (float): Координата Y начальной точки (по умолчанию 50000, середина C5)
        z (float): Координата Z начальной точки (по умолчанию 3500, средняя глубина)
        num_electrodes (int): Количество электродов (по умолчанию 1)
        spacing (float): Интервал между электродами в микрометрах (по умолчанию 1000)
        axis (str): Ось, вдоль которой располагаются электроды: "x", "y" или "z" (по умолчанию "y")

    Returns:
        list: Список координат электродов в формате [(x1, y1, z1), (x2, y2, z2), ...]
    """
    # Проверяем входные параметры
    if num_electrodes < 1:
        raise ValueError("Параметр num_electrodes должен быть >= 1")
    if spacing < 0:
        raise ValueError("Параметр spacing должен быть >= 0")
    if axis not in ["x", "y", "z"]:
        raise ValueError("Параметр axis должен быть 'x', 'y' или 'z'")

    # Создаём список координат
    electrode_positions = []

    for i in range(num_electrodes):
        if axis == "x":
            pos = (x + i * spacing, y, z)
        elif axis == "y":
            pos = (x, y + i * spacing, z)
        else:  # axis == "z"
            pos = (x, y, z + i * spacing)
        electrode_positions.append(pos)

    print(f"Заданы координаты {num_electrodes} электродов с интервалом {spacing} вдоль оси {axis}:")
    for i, pos in enumerate(electrode_positions):
        print(f"Электрод {i+1}: {pos}")

    return electrode_positions

def run_simulations(netParams, num_runs=1, sim_duration=1000, record_step=0.025, save_formats=None):
    """
    Запускает несколько симуляций и управляет наборами данных.

    Args:
        netParams: Параметры сети (объект NetParams)
        num_runs (int): Количество прогонов (размер набора данных), по умолчанию 1
        sim_duration (float): Длительность симуляции (в мс), по умолчанию 1000 мс
        record_step (float): Шаг записи данных (в мс), по умолчанию 0.025 мс
        save_formats (list): Форматы для сохранения данных (например, ['json', 'pickle']), по умолчанию ['json', 'pickle']
    """
    if save_formats is None:
        save_formats = ['json', 'pickle']

    # Проверяем, что указанные форматы поддерживаются (CSV обрабатывается в calculate_electrode_signal)
    supported_formats = ['json', 'pickle']
    for fmt in save_formats:
        if fmt not in supported_formats:
            raise ValueError(f"Формат {fmt} не поддерживается в run_simulations. Доступные форматы: {supported_formats}. Для CSV используйте calculate_electrode_signal.")

    # Запускаем симуляции
    for run_idx in range(num_runs):
        print(f"Запуск симуляции {run_idx + 1} из {num_runs}...")

        # Настройка параметров записи данных
        simConfig = specs.SimConfig()

        # Задаём длительность симуляции и шаг времени
        simConfig.duration = sim_duration
        simConfig.dt = 0.025

        # Параметры записи напряжений (только V_soma)
        simConfig.recordTraces = {
            'V_soma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}
        }

        # Записываем данные со всех нейронов
        simConfig.recordCells = ['all']
        simConfig.recordStep = record_step

        # Настраиваем имя файла для сохранения данных
        simConfig.filename = f'simulation_run_{run_idx + 1}'

        # Опции для сохранения данных
        simConfig.saveJson = 'json' in save_formats
        simConfig.savePickle = 'pickle' in save_formats
        simConfig.saveDataInclude = ['simData', 'simConfig', 'netParams', 'net']
        simConfig.recordStim = True  # Включаем запись стимуляций
        # Настройка анализа
        # simConfig.analysis['plot2Dnet'] = {'saveFig': False}

        # # Задаём сиды для воспроизводимости
        # base_seed = 12345 + run_idx * 1000
        # simConfig.seeds = {
        #     'conn': base_seed,
        #     'loc': base_seed + 1,
        #     'stim': base_seed + 2
        # }
        # from neuron import h
        # h.Random().Random123_globalindex(base_seed)

        # Запускаем симуляцию
        sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)

        # Закрываем графики
        plt.close('all')

    print(f"Все симуляции завершены. Данные сохранены в форматах: {save_formats}")

def calculate_electrode_signal(electrode_positions, power=2, save_to_csv=False, run_idx=1, pain_info=None, trim_start_ms=100):

    # Проверяем, что симуляция была выполнена
    if not hasattr(sim, 'allSimData'):
        raise ValueError("Симуляция не выполнена. Сначала запустите sim.simulate()")

    # Проверяем, что мембранный потенциал записан
    if 'V_soma' not in sim.allSimData:
        raise ValueError("Мембранный потенциал не записан. Добавьте 'V_soma' в simConfig.recordTraces")

    # Получаем данные о нейронах
    #cell_positions = sim.net.getCellPositions()  # Позиции всех нейронов [(x, y, z), ...]
    # Получаем данные о нейронах
    cell_positions = []
    for cell in sim.net.cells:
        # Извлекаем координаты из tags или secs
        x = cell.tags.get('x', 0)
        y = cell.tags.get('y', 0)
        z = cell.tags.get('z', 0)
        cell_positions.append((x, y, z))
    v_data = sim.allSimData['V_soma']  # Мембранный потенциал всех нейронов

    # Временные шаги симуляции
    t = np.array(sim.allSimData['t'])
    num_steps = len(t)

    # Инициализируем сигнал для каждого электрода
    signals = {i: np.zeros(num_steps) for i in range(len(electrode_positions))}

    # Обрабатываем каждый электрод
    for elec_idx, elec_pos in enumerate(electrode_positions):
        elec_x, elec_y, elec_z = elec_pos

        # Учитываем вклад всех нейронов
        for gid, cell_pos in enumerate(cell_positions):
            cell_x, cell_y, cell_z = cell_pos[0], cell_pos[1], cell_pos[2]  # Координаты нейрона
            distance = np.sqrt((elec_x - cell_x)**2 + (elec_y - cell_y)**2 + (elec_z - cell_z)**2)

            # Избегаем деления на ноль (если электрод в точке нейрона)
            if distance < 1e-6:
                distance = 1e-6

            # Вычисляем вес на основе расстояния
            weight = 1 / (distance ** power)

            # Добавляем вклад мембранного потенциала
            v_trace = v_data[f'cell_{gid}']
            for t_idx in range(num_steps):
                signals[elec_idx][t_idx] += v_trace[t_idx] * weight
        # Сохранение в CSV
        if save_to_csv:
            # Создаём бинарную колонку для боли
            pain_values = np.zeros(len(t))
            if pain_info:
                for stim_name, info in pain_info.items():
                    start_time = info.get('start', 0)
                    duration = info.get('duration', 0)
                    end_time = start_time + duration
                    pain_mask = (t >= start_time) & (t <= end_time)
                    pain_values[pain_mask] = 1

            # Обрезаем первые trim_start_ms миллисекунд
            mask = t >= trim_start_ms
            t_trimmed = t[mask]
            signal_trimmed = signals[elec_idx][mask]
            pain_values_trimmed = pain_values[mask]
            # Округляем время до 6 знаков после запятой
            t_trimmed = np.round(t_trimmed, decimals=6)
            # Создаём DataFrame с тремя столбцами
            df = pd.DataFrame({
                'time_ms': t_trimmed,
                'electrode_signal_volts': signal_trimmed,
                'pain': pain_values_trimmed
            })
            df.to_csv(f'electrode_{elec_idx}_run_{run_idx}.csv', index=False)
    return signals

def plot_electrode_pain_signals(electrode_positions, signals, pain_info=None, trim_start_ms=100):

    # Рисуем сигналы для каждого электрода
    for elec_idx, (elec_pos, signal) in enumerate(zip(electrode_positions, signals.values())):

        t = np.array(sim.allSimData.get('t', []))
        signal = signals.get(elec_idx, np.zeros(len(t)))
        # Обрезаем первые trim_start_ms миллисекунд
        mask = t >= trim_start_ms
        t_trimmed = t[mask]
        signal_trimmed = signal[mask]
        plt.figure(figsize=(10, 6))
        plt.plot(t_trimmed, signal_trimmed, label=f"Электрод {elec_idx} ({elec_pos[0]}, {elec_pos[1]}, {elec_pos[2]})")
        # Отмечаем интервалы боли
        if pain_info:
            for pop_name, info in pain_info.items():
                start_time = info.get('start', 0)
                duration = info.get('duration', 0)
                end_time = start_time + duration + 100
                # Вертикальные линии для начала и конца боли
                plt.axvline(x=start_time, color='red', linestyle='--', alpha=0.5, label=f'Начало боли ({pop_name})' if elec_idx == 0 else "")
                plt.axvline(x=end_time, color='green', linestyle='--', alpha=0.5, label=f'Конец боли ({pop_name})' if elec_idx == 0 else "")
                # Альтернатива: заливка интервала
                # plt.axvspan(start_time, end_time, alpha=0.1, color='red', label=f'Боль ({pop_name})' if elec_idx == 0 else "")

                plt.xlabel('Время (мс)')
                plt.ylabel('Сигнал электрода (мВ)')
                plt.title('Сигналы на электродах с интервалами боли')
                plt.legend()
                plt.grid(True)

       
        plt.savefig(f'signals_pain_{elec_idx}')
        plt.show()

