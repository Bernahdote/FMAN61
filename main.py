from func_gn import get_data_json, phi0, phi1, phi2
from gauss_newton import gauss_newton
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__": 
   
   func = lambda x, t: phi2(x,t)
   t, y = get_data_json("data2.json")
   x0 = np.array([4.,3.,2.,1.])
   x, N_eval, N_iter, normg = gauss_newton(func, t, y, x0, 1.e-4, True, True)

   print("Solution: ", x)
   plt.plot(t, y, 'ro', label='data points')
   plt.plot(t, func(x, t), 'b-', label='fitted curve')
   plt.show()

    




