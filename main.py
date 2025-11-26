
from func_gn import phi0, phi1, phi2, get_data_json
from grad import grad_c, jacobian_c
import numpy as np




if __name__ == "__main__": 

    t_data, y_data = get_data_json() 
    c = np.array([1.0, 2.0])

    max_iter = 100 

    for k in range(max_iter): 
        y_pred = phi0(c, t_data)
        r = y_pred - y_data

        g = grad_c(c, r)
        J = jacobian_c(c, t_data)

        A = J.T @ J
        delta_c = -np.linalg.solve(A, g)

    




