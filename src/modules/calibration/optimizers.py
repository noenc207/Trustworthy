import numpy as np
from typing import Callable, Tuple, Dict, Any
from scipy.optimize import minimize
import time

def optimize_lbfgs(
    objective_fn: Callable[[np.ndarray], float],
    initial_temp: float = 1.5,
    bounds: Tuple[float, float] = (0.1, 10.0),
    max_iter: int = 1000
) -> Dict[str, Any]:
    """Optimizes temperature using L-BFGS-B."""
    start_time = time.time()
    
    def wrapped_obj(t):
        return objective_fn(t)
        
    res = minimize(
        wrapped_obj, 
        x0=np.array([initial_temp]), 
        bounds=[bounds], 
        method='L-BFGS-B',
        options={'maxiter': max_iter}
    )
    
    exec_time = time.time() - start_time
    
    return {
        "temperature": float(res.x[0]) if res.success else initial_temp,
        "success": bool(res.success),
        "iterations": int(res.nit),
        "final_loss": float(res.fun),
        "execution_time": exec_time,
        "message": str(res.message)
    }

def optimize_adam_numpy(
    objective_fn: Callable[[np.ndarray], float],
    initial_temp: float = 1.5,
    bounds: Tuple[float, float] = (0.1, 10.0),
    lr: float = 0.01,
    max_iter: int = 1000,
    patience: int = 10,
    min_improvement: float = 1e-4
) -> Dict[str, Any]:
    """Custom NumPy implementation of Adam optimizer for Temperature Scaling."""
    start_time = time.time()
    
    t = initial_temp
    
    # Adam parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m = 0.0
    v = 0.0
    
    best_loss = float('inf')
    best_t = t
    patience_counter = 0
    
    # Gradient approximation step
    eps_grad = 1e-4
    
    final_loss = best_loss
    iterations = 0
    
    for i in range(1, max_iter + 1):
        iterations = i
        
        # Current loss
        loss = objective_fn(np.array([t]))
        
        # Central difference gradient approximation
        loss_plus = objective_fn(np.array([t + eps_grad]))
        loss_minus = objective_fn(np.array([t - eps_grad]))
        grad = (loss_plus - loss_minus) / (2 * eps_grad)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        m_hat = m / (1 - beta1 ** i)
        v_hat = v / (1 - beta2 ** i)
        
        t_new = t - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Apply bounds
        t = float(np.clip(t_new, bounds[0], bounds[1]))
        
        # Early stopping check
        if loss < best_loss - min_improvement:
            best_loss = loss
            best_t = t
            patience_counter = 0
        else:
            patience_counter += 1
            
        final_loss = loss
            
        if patience_counter >= patience:
            break
            
    exec_time = time.time() - start_time
    
    return {
        "temperature": best_t,
        "success": True,
        "iterations": iterations,
        "final_loss": best_loss,
        "execution_time": exec_time,
        "message": "Adam optimization completed"
    }
