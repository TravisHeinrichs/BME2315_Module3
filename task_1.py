#task one
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = pd.read_csv("cavalitide_data.csv") #obtain the data

cavalitide_conc = data["L2_nM"].to_numpy() #assign variables to columns of experimental data
exp_F_act = data["Fact"].to_numpy()

def find_F_act(cavalitide_conc, Kd_C): #gives the fraction of active receptors using inputted Cavalitide concetration and its Kd, Kd_G (0.17 [nM]) and GLP1 concentration (0.015 nM) are set as constant
    return ((0.015/0.17) + (cavalitide_conc/Kd_C))/ (1 + (0.015/0.17) + (cavalitide_conc/Kd_C))

popt, pcov = curve_fit(find_F_act, cavalitide_conc, exp_F_act, p0 = [0.1]) #initial guess is Kd_C = 0.1

Kd_C = popt[0]

plt.figure()
plt.semilogx(cavalitide_conc, exp_F_act, "o", label = "Experimental Data")
plt.xlabel("Cavalitide Concentration (nM)")
plt.ylabel("Fration of Active Receptors")
plt.semilogx(cavalitide_conc, find_F_act(cavalitide_conc, Kd_C), label = "Fitted Model (Using Steepest Descent)")
plt.legend()
plt.show()

print("Fitted Kd for Cavalitide:", Kd_C, "nM") #returns 0.077 nM
#As the Kd of Semaglutide is 0.350 nM and the Kd of Cavalitide is 0.077 nM, Cavalitide has over 4.5 times the binding affinity to the GLP-1 receptor!
#The advantage of Cavalitide is that it takes less dose to achieve the same receptor activation. Maybe this brings the cost down!
#The disadvantage is that the higher binding affinity may mean Cavalitide has greater side effects or the body might not be able to eliminate it for an undesired time.

