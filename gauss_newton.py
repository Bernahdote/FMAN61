from typing import Callable, Tuple
import numpy as np
from grad import grad_c, jacobian_c

def gauss_newton(phi : Callable[[np.ndarray, np.ndarray], np.ndarray], t:  np.ndarray, y : np.ndarray, x0 : np.ndarray, tol: float, printout: bool, plotout : bool) -> Tuple[np.ndarray, int, int, float]:
    max_iter = 100

    xk = x0.copy()

    def resid(x_):
        return y - phi(x_, t)
    def r_norm(x_):
            return np.dot(resid(x_), resid(x_))
        

    max_iter = 10000
    for i in range(max_iter): 
        r = resid(xk)
        #J = jacobian_c(lambda x: phi(x,t), x)
        J = jacobian_c(resid, xk)
        grad_f = -J.T @ r
        A = J.T @ J

        try:
            d_k = np.linalg.solve(A, grad_f)
        except np.linalg.LinAlgError as e:
            d_k = np.linalg.solve(A + 1.e-8*np.eye(A.shape[0]), grad_f)
            #d_k = np.linalg.lstsq(A, grad_f, rcond=None)[0]
        slope = grad_f @ d_k

        # armijo line search
        alpha = 2
        lbda = 1.0 
        while(r_norm(xk + alpha*lbda*d_k) < r_norm(xk) + 0.1 *lbda* alpha * slope):
            lbda = lbda*alpha
    
        while(r_norm(xk + lbda*d_k) > r_norm(xk) + 0.1 * lbda*slope):
            lbda = lbda / alpha

        xk = xk + lbda*d_k # iterate

    
    return tuple((xk, max_iter, 0, 0.0))

        
    

        

   















