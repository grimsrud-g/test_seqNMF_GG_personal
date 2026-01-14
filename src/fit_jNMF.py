import os
import numpy as np
import sys
sys.path.append('..')
from .models import jSeqNMF
from tqdm import tqdm

import torch


def _best_checkpoint_path(result_file):
    base, ext = os.path.splitext(result_file)
    if not ext:
        return result_file + '_best'
    return f"{base}_best{ext}"


def _infer_best_from_history(cost, cur_iter):
    if cur_iter < 0:
        return -1, float('inf')
    history = cost[:cur_iter + 1]
    if history.size == 0:
        return -1, float('inf')
    valid_mask = np.isfinite(history) & (history > 0)
    if not np.any(valid_mask):
        return -1, float('inf')
    valid_indices = np.arange(history.size)[valid_mask]
    best_idx = np.argmin(history[valid_mask])
    best_iter = int(valid_indices[best_idx])
    best_cost = float(history[valid_mask][best_idx])
    return best_iter, best_cost


def _save_checkpoint(path, W, H_list, cost, iters, r2_list, best_iter, best_cost,
                     checkpoint_type):
    results = {
        'W': W,
        'cost': cost,
        'iters': iters,
        'r2_list': r2_list,
        'best_iter': best_iter,
        'best_cost': best_cost,
        'checkpoint_type': checkpoint_type,
    }
    for idx, H in enumerate(H_list):
        results[f'H{idx}'] = H
    np.savez(path, **results)

def fit_jNMF(data_loader, result_file, K, L, lam, lambda_OrthH, lambda_OrthW, lambda_L1W, lambda_L1H, nH, seed, num_iters, device, dtype=torch.float32, no_improve_tol=50, W_fixed=False, W=None, renorm_type='renorm'):
    assert ((not W_fixed) and (W is None)) or (W_fixed and (W is not None)), "W must be provided if W_fixed is True"
    with torch.no_grad():
        torch.manual_seed(seed)
        np.random.seed(seed)

        best_result_file = _best_checkpoint_path(result_file)

        try:
            result = np.load(result_file)
            cost = result['cost']
            if len(cost) > num_iters:
                print('Already fitted')
                return
            elif len(cost) < num_iters:
                cost = np.concatenate((cost, np.zeros(num_iters - len(cost))))
            cur_iter = int(result['iters'])

            if 'best_iter' in result:
                best_iter = int(result['best_iter'])
            else:
                best_iter = None

            if 'best_cost' in result:
                best_cost = float(result['best_cost'])
            else:
                best_cost = None

            if (best_iter is None or best_cost is None or not np.isfinite(best_cost)) and os.path.exists(best_result_file):
                with np.load(best_result_file) as best_result:
                    if 'best_cost' in best_result:
                        best_cost = float(best_result['best_cost'])
                    if 'iters' in best_result:
                        best_iter = int(best_result['iters'])

            if best_iter is None or best_cost is None or not np.isfinite(best_cost):
                inferred_iter, inferred_cost = _infer_best_from_history(cost, cur_iter)
                best_iter = inferred_iter
                best_cost = inferred_cost

            if best_iter < 0 or not np.isfinite(best_cost):
                best_cost = float('inf')
                best_iter = -1

            no_improve = cur_iter - best_iter if best_iter >= 0 else 0

            if no_improve == no_improve_tol or cur_iter == num_iters - 1:
                return

            W = result['W']
            H = [result[f'H{iH}'] for iH in range(nH)]
            
            model = jSeqNMF(
                K=K, L=L, lam=lam, nH=nH, device=device,
                lambda_OrthH=lambda_OrthH, lambda_OrthW=lambda_OrthW,
                lambda_L1H=lambda_L1H, lambda_L1W=lambda_L1W,
                W_init=W, H_init=H, W_fixed=W_fixed,
                dtype=dtype
            )
            
            print('loaded')
            
        except (FileNotFoundError, EOFError, NameError):
            model = jSeqNMF(
                K=K, L=L, lam=lam, nH=nH, device=device,
                lambda_OrthH=lambda_OrthH, lambda_OrthW=lambda_OrthW,
                lambda_L1H=lambda_L1H, lambda_L1W=lambda_L1W, 
                W_init=W, W_fixed=W_fixed,
                dtype=dtype
            )
            cost = np.zeros((num_iters))
            cur_iter = 0
            best_iter = -1
            best_cost = float('inf')
            no_improve = 0

            for batch in data_loader:
                Xt, Mt, iH = batch
                Xt = Xt.to(device=device)
                Mt = Mt.to(device=device)

                if not model.H_initialized:
                    model.initialize_H(Xt, iH)
            if not model.W_initialized:
                model.initialize_W(Xt)

        W_numerator = torch.zeros(model.W.shape, device=device, dtype=dtype)
        W_denominator = torch.zeros(model.W.shape, device=device, dtype=dtype)
        pbar = tqdm(range(cur_iter+1, num_iters), desc="Fitting jNMF")
        for i in pbar:
            is_new_best = False
            W_numerator *= 0
            W_denominator *= 0
            r2_list = [list() for _ in range(nH)]
            for batch in data_loader:
                Xt, Mt, iH = batch
                Xt = Xt.to(device=device)
                Mt = Mt.to(device=device)
                iH = iH.item()

                res = model.do_mult_update_step(Xt, iH, Mt)
                if not W_fixed:
                    W_numerator += res[0]
                    W_denominator += res[1]
                c, r2 = model.compute_cost(Xt, iH, Mt)
                cost[i] += c
                r2_list[iH] = r2
                
                del Xt, Mt, res, c, r2

            # update W
            eps = torch.finfo(dtype).eps
            if not W_fixed:
                model.W *= W_numerator / (W_denominator + eps)
            if renorm_type == 'renorm':
                model.renorm_epoch()
            elif renorm_type == 'renorm_W_norm':
                model.renorm_epoch_based_on_W_norm()
            elif renorm_type == 'renorm_H_norm':
                model.renorm_epoch_based_on_H_norm()
            else:
                raise ValueError(f"Unknown renorm_type: {renorm_type}")
            if cost[i] < best_cost:
                best_iter = i
                best_cost = cost[i]
                no_improve = 0
                is_new_best = True
            else:
                no_improve += 1

            if i % 10 == 0 or i == num_iters - 1 or no_improve == no_improve_tol:
                W, H = model.finalize(return_xhat=False)
                _save_checkpoint(
                    result_file,
                    W,
                    H,
                    cost,
                    i,
                    r2_list,
                    best_iter,
                    best_cost,
                    'current'
                )
                if is_new_best:
                    _save_checkpoint(
                        best_result_file,
                        W,
                        H,
                        cost,
                        i,
                        r2_list,
                        best_iter,
                        best_cost,
                        'best'
                    )
                if no_improve == no_improve_tol:
                    break

                model = jSeqNMF(
                    K=K, L=L, lam=lam, nH=nH, device=device,
                    lambda_OrthH=lambda_OrthH, lambda_OrthW=lambda_OrthW,
                    lambda_L1H=lambda_L1H, lambda_L1W=lambda_L1W,
                    W_init=W, H_init=H, W_fixed=W_fixed,
                    dtype=dtype
                )
                
                del W, H
                
            # Release unnecessary GPU memory
            torch.cuda.empty_cache()
                
            pbar.set_description(f"Fitting jNMF | Epoch: {i}/{num_iters} | Cost: {np.round(cost[i], 3)} | Best Cost: {np.round(best_cost, 3)} | R2: {np.round(r2_list, 3)[:3]} | No Improve: {no_improve}")
            pbar.update(1)
        pbar.close()
