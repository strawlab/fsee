import fsee.EMDSim
import matplotlib.pyplot as plt
import numpy as np

def baylor_eqn(t, t_p=0.156*0.75, r=1.0, n=4):

    # equation 2 in Kraft, T.W., Schneeweis, D.M., and Schnapf,
    # J.L. (1993) Visual Transduction in Human Rod
    # Photoreceptors. Journal of Physiology. 464, p 747-765.

    # Default values from caption of Fig. 3, then scaled by 75% to
    # approximate light adapted values as suggested in the "Kinetics
    # of flash responses on steady backgrounds" section.

    tbar = t/t_p
    j = r * (tbar*np.exp(1-tbar))**(n-1)
    return j

t = np.linspace(0,1,1000)
fly = fsee.EMDSim.log_normal(t)
human = baylor_eqn(t)

plt.plot(t,fly,label='fly')
plt.plot(t,human,label='human')
plt.legend()
plt.gca().text(0,0,'BAD Comparison. Better to use Hateren, Ruettiger, Sun, Lee, (2002) J. Neurosci. macaque data')
plt.show()
