################################################################################
#                                                                              #
#  █████   ██████  █████   ██  ██████                                          #
#  ██   ██ ██  ██  ██   ██ ██  ██  ██                                          #
#  █████   ██  ██  █████   ██  ██  ██                                          #
#  ██   ██ ██  ██  ██   ██ ██  ██  ██                                          #
#  █████   ██████  █████   ██  ██████                                          #
#                                                                              #
#  Author: Adrian Martens                                                      #
#  Project: BOBIO - Bayesian Optimizaition for Bioprocesses                    #
#  Date: November 2024                                                         #
#                                                                              #
################################################################################

import pandas as pd

# Dictionary-structure of process parameters

# Y_P_X = 1.495 mg/(10^6 cells * d) ---> Y_P_X = 1.495/(24*10^6) mg/(cell*h)
# Y_dot_P_X = 4.41 mg/(10^6 cells) ---> Y_dot_P_X = 4.41/(10^6) mg/(cell)

reactor_list = ["3LBATCH", "3LCONTBATCH", "15LCONTBATCH"]

process_parameters = {
    "3LBATCH": {
        "celltype_1": {"my_max": 0.035, "K_lysis": 4e-2,   "k": [1e-3, 1e-2, 1e-2],          "K": [150, 40, 1, 0.22],    "Y": [9.23e7, 8.8e8, 1.6, 0.68, 6.2292e-8, 4.41e-6],    "m": [8e-13, 3e-12], "A": 1e1, "pH_opt": 7.2, "E_a": 32},
        
    },
    "3LCONTBATCH": {
        "celltype_1": {"my_max": 0.035, "K_lysis": 4e-2,   "k": [1e-3, 1e-2, 1e-2],          "K": [150, 40, 1, 0.22],    "Y": [9.23e7, 8.8e8, 1.6, 0.68, 6.2292e-8, 4.41e-6],    "m": [8e-13, 3e-12], "A": 1e1, "pH_opt": 7.2, "E_a": 32},

    },
    "15LCONTBATCH": {
        "celltype_1": {"my_max": 0.035, "K_lysis": 4e-2,   "k": [1e-3, 1e-2, 1e-2],          "K": [150, 40, 1, 0.22],    "Y": [9.23e7, 8.8e8, 1.6, 0.68, 6.2292e-8, 4.41e-6],    "m": [8e-13, 3e-12], "A": 1e1, "pH_opt": 7.2, "E_a": 32},
        
    }
}

noise_level = {
            "3LBATCH": 2e-1,    
            "3LCONTBATCH": 8e-2,
            "15LCONTBATCH": 8e-5
        }

fidelity_cost = {
            "3LBATCH": 0.05,    
            "3LCONTBATCH": 0.5,
            "15LCONTBATCH": 1
        }

# Erstelle eine Liste der Kombinationen
data = []
for reactor, cell_data in process_parameters.items():
    for cell_type, params in cell_data.items():
        entry = {
            "reactor": reactor,
            "cell_type": cell_type,
            **params
        }
        data.append(entry)

# Konvertiere die Liste in einen DataFrame
df = pd.DataFrame(data)