import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqNMF(nn.Module):
    def __init__(self,
        K, L, lam,
        W_init=None, H_init=None,
        max_iter=100, tol=-float("Inf"), 
        shift=True, sort_factors=True,
        lambda_L1W=0, lambda_L1H=0, 
        lambda_OrthH=0, lambda_OrthW=0, 
        M=None, use_W_update=True, W_fixed=False,
        device = torch.device('cuda'),
        dtype=torch.float32
        ):

        super(SeqNMF, self).__init__()
        self.K = K
        self.L = L
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.shift = shift
        self.sort_factors = sort_factors
        self.lambda_L1W = lambda_L1W
        self.lambda_L1H = lambda_L1H
        self.lambda_OrthH = lambda_OrthH
        self.lambda_OrthW = lambda_OrthW
        self.M = M
        if self.M is not None:
            self.M = np.pad(self.M, ((0,0), (self.L, self.L)))

        self.use_W_update = use_W_update
        self.W_fixed = W_fixed
        self.device = device
        self.H_initialized = False
        self.W_initialized = False

        self.dtype = dtype

        if W_init is not None:
            self.N = W_init.shape[0]
            self.W = nn.Parameter(torch.Tensor(self.N, self.K, self.L))
            #if not type(W_init) == torch.Tensor:
            W_init = torch.tensor(
                    W_init, device=self.device, dtype=dtype)

            self.W.data = W_init
            self.W_initialized = True
        
        if H_init is not None:
            H_init = np.pad(H_init, ( (0,0), (self.L, self.L) ))
            self.T = H_init.shape[1]
            self.H = nn.Parameter(torch.Tensor(self.K, self.T))
            #if not type(H_init) == torch.Tensor:
            H_init = torch.tensor(
                    H_init, device=self.device, dtype=dtype)

            self.H.data = H_init
            self.H_initialized = True

    def initialize_W(self, X):
        self.N = X.shape[1] # X is unsqueeze(0)'d here
        self.W = nn.Parameter(torch.Tensor(self.N, self.K, self.L))
        self.W.data = X.max()*torch.rand(self.N, self.K, self.L, dtype=self.dtype).to(self.device) / np.sqrt(X.shape[-1]/3)

    def initialize_H(self, X):
        self.T = X.shape[-1]
        self.H = nn.Parameter(torch.Tensor(self.K, self.T))
        self.H.data = X.max()*torch.rand(self.K, self.T, dtype=self.dtype).to(self.device) / np.sqrt(self.T/3)
        
    def get_padWrev(self):
        return F.pad(torch.flip(self.W, (2,)), (1,0))
    
    def forward(self):
        padH = F.pad(self.H, (self.L-1, 0))
        Wrev = torch.flip(self.W, (2,))
        X_hat = F.conv1d(padH.unsqueeze(0), Wrev, groups=1, padding='valid')
        return X_hat
    
    def get_WTX(self, X):
        padX = F.pad(X, (0, self.L-1))
        WTX = F.conv1d(
                padX.unsqueeze(0), 
                torch.swapdims(self.W, 0, 1), 
                padding='valid')
        return WTX
    
    def get_dRdH(self, WTX, smooth_kernel):
        WTXS = F.conv1d(
            WTX,
            torch.tile(smooth_kernel, (self.K, 1)).unsqueeze(1),
            padding='same', groups=self.K).squeeze(0)
        dRdH = self.lam * (1-torch.eye(self.K, device=self.device, dtype=self.dtype)) @ WTXS
        return dRdH
    
    def get_dHHdH(self, smooth_kernel):
        HS = F.conv1d(
            self.H.unsqueeze(0), 
            torch.tile(smooth_kernel, (self.K, 1)).unsqueeze(1), 
            padding='same', groups=self.K).squeeze(0)
        HSH_inej = (1-torch.eye(self.K, device=self.device, dtype=self.dtype)) @ HS
        return self.lambda_OrthH * HSH_inej
    
    def get_Wflat(self):
        return self.W.sum(-1)
    
    def renormalize(self):
        norms = torch.norm(self.H,2,dim=1)
        self.H.data = torch.diag( 1 / ( norms + torch.finfo(self.H.dtype).eps ) ) @ self.H
        self.W.data *= torch.tile( norms[None,:,None], (self.N, 1, self.L) )
    
    def get_XS(self, X, smooth_kernel):
        XS = F.conv1d(
            X,
            torch.tile(smooth_kernel, (self.N, 1)).unsqueeze(1), 
            padding='same', groups=self.N).squeeze()
        return XS
    
    def finalize(self):
        Xhat = self.forward().squeeze().to('cpu').detach().numpy()
        self.H.data = self.H.data[:,self.L:-self.L]
        return Xhat[:, self.L:-self.L], self.W.data.to('cpu').detach().numpy(), self.H.data.to('cpu').detach().numpy()

    def shift_factors(self):
        W = self.W
        H = self.H

        Wpad = F.pad(W, (self.L, self.L))
        if self.L > 1:
            center = self.L // 2
            Wmass = W.sum(0)
            com = torch.floor( (torch.arange(1, self.L+1, device=self.device) * Wmass ).sum(1) / Wmass.sum(1) )
            bad_idx = torch.isnan(com)
            com = com.type(torch.IntTensor)
            for idx in torch.where(~bad_idx)[0]:
                Wpad[:,idx,:] = torch.roll(Wpad[:,idx,:], (center - com[idx].item(),), dims=1)
                H[idx,:] = torch.roll(H[idx,:], (com[idx].item() - center,), dims=0)

        self.W.data = Wpad[:,:,self.L:-self.L]
        self.H.data = H

    def compute_cost(self, X, Mt):

        Mt = Mt[0, 0, self.L:-self.L].cpu().numpy()
        Xhat = self.forward()[...,self.L:-self.L][0].detach().cpu().numpy()[:, ~Mt]
        Xtrm = X[...,self.L:-self.L][0].detach().cpu().numpy()[:, ~Mt]
        cost = np.sqrt(((Xtrm - Xhat)**2).sum())
        r2 = np.mean(1 - ((Xtrm - Xhat) ** 2).sum(-1) / ((Xtrm - Xtrm.mean(-1, keepdims=True)) ** 2 + 1e-8).sum(-1))
        return cost, r2

    def compute_xortho(self, X):
        Xhat = self.forward()[..., self.L:-self.L]
        WTX = self.get_WTX(X.squeeze())
        smooth_kernel = torch.ones(1, 2*self.L - 1).to(self.device)
        WTXS = F.conv1d(
            WTX,
            torch.tile(smooth_kernel, (self.K, 1)).unsqueeze(1),
            padding='same', groups=self.K).squeeze(0)
        WTXSHT = (WTXS @ self.H.T) @ ( 1 - torch.eye(self.K, device=self.device, dtype=self.dtype) )
        reg = torch.linalg.norm(WTXSHT)
        return reg

    def do_mult_update_step(self, X, Mt):
        self.M = Mt
        smallnum = X.max()*1e-6
        eps = torch.finfo(X.dtype).eps

        smooth_kernel = torch.ones(1, 2*self.L - 1, dtype=self.dtype).to(self.device) / (2*self.L - 1)
        X_hat = self.forward()
        
        WTX = self.get_WTX(X.squeeze())
        WTXhat = self.get_WTX(X_hat.squeeze())

        if self.lam > 0:
            dRdH = self.get_dRdH(WTX, smooth_kernel)
            
        else:
            dRdH = 0
            
        if self.lambda_OrthH > 0:
            dHHdH = self.get_dHHdH(smooth_kernel)

        else:
            dHHdH = 0
        
        dRdH = dRdH + self.lambda_L1H + dHHdH
        
        # update H
        self.H *= (WTX / (WTXhat + dRdH + eps)).squeeze(0)
        
        # Shift to center factors
        self.shift_factors()
        self.W += smallnum
        
        # Renormalize so rows of H have constant energy
        self.renormalize()
        
        # The updated implementation so the mask works.
        if not self.W_fixed:
            Xhat = self.forward()            
            if self.M is not None:
                # mask = torch.from_numpy(self.M)
                X[self.M] = Xhat[self.M]

            if self.lambda_OrthW > 0:
                Wflat = self.W.sum(-1)
            
            if (self.lam > 0) & self.use_W_update:
                XS = self.get_XS(X, smooth_kernel)
            
            for l in range(self.L):
                H_shifted = torch.roll(self.H, l, dims=1)
                XHT = X @ H_shifted.T
                XhatHT = Xhat @ H_shifted.T
                
                if self.lam > 0 & self.use_W_update:
                    dRdW = self.lam * XS @ H_shifted.T @ ( 1 - torch.eye(self.K, device=self.device, dtype=self.dtype) )
                else:
                    dRdW = 0
                    
                if self.lambda_OrthW > 0:
                    dWWdW = self.lambda_OrthW * Wflat @ ( 1 - torch.eye(self.K, device=self.device, dtype=self.dtype) )
                else:
                    dWWdW = 0
                    
                dRdW += self.lambda_L1W + dWWdW
                
                self.W[...,l] *= (XHT / (XhatHT + dRdW + eps)).squeeze(0)

class jSeqNMF(nn.Module):
    def __init__(self,
        K, L, lam, nH, 
        W_init=None, H_init=None,
        max_iter=100, tol=-float("Inf"), 
        shift=True, sort_factors=True,
        lambda_L1W=0, lambda_L1H=0, 
        lambda_OrthH=0, lambda_OrthW=0, 
        use_W_update=True, W_fixed=False,
        device = torch.device('cuda'),
        dtype=torch.float32
        ):

        super(jSeqNMF, self).__init__()
        self.K = K
        self.L = L
        self.lam = lam
        self.nH = nH
        self.max_iter = max_iter
        self.tol = tol
        self.shift = shift
        self.sort_factors = sort_factors
        self.lambda_L1W = lambda_L1W
        self.lambda_L1H = lambda_L1H
        self.lambda_OrthH = lambda_OrthH
        self.lambda_OrthW = lambda_OrthW

        self.use_W_update = use_W_update
        self.W_fixed = W_fixed
        self.device = device
        self.H_initialized = False
        self.W_initialized = False

        self.dtype = dtype

        self.H = [list() for _ in range(self.nH)]

        if W_init is not None:
            self.N = W_init.shape[0]
            self.W = nn.Parameter(torch.Tensor(self.N, self.K, self.L))
            #if not type(W_init) == torch.Tensor:
            W_init = torch.tensor(
                    W_init, device=self.device, dtype=self.dtype)

            self.W.data = W_init
            self.W_initialized = True
        
        if H_init is not None:
            # shape: nH * K * T
            assert len(H_init) == self.nH
            for iH in range(self.nH):
                H_ = np.pad(H_init[iH], ( (0,0), (self.L, self.L) ))
                self.T = H_.shape[-1]
                self.H[iH] = nn.Parameter(torch.Tensor(self.K, self.T))
                #if not type(H_init) == torch.Tensor:
                H_ = torch.tensor(
                        H_, device=self.device, dtype=self.dtype)

                self.H[iH].data = H_
            self.H_initialized = True

    def initialize_W(self, X):
        self.N = X.shape[1] # X is unsqueeze(0)'d here
        self.W = nn.Parameter(torch.Tensor(self.N, self.K, self.L))
        self.W.data = X.max()*torch.rand(self.N, self.K, self.L, dtype=self.dtype).to(self.device)

    def initialize_H(self, X, iH):
        self.T = X.shape[-1]
        self.H[iH] = nn.Parameter(torch.Tensor(self.K, X.shape[-1]))
        self.H[iH].data = X.max()*torch.rand(self.K, X.shape[-1], dtype=self.dtype).to(self.device) / np.sqrt(X.shape[-1]/3)
        
    def get_padWrev(self):
        return F.pad(torch.flip(self.W, (2,)), (1,0))
    
    def forward(self, iH):
        # iH means the ith dataset
        padH = F.pad(self.H[iH], (self.L-1, 0))
        Wrev = torch.flip(self.W, (2,))
        X_hat = F.conv1d(padH, Wrev, groups=1, padding='valid')
        return X_hat.unsqueeze(0)
    
    def get_WTX(self, X):
        padX = F.pad(X, (0, self.L-1))
        WTX = F.conv1d(
                padX.unsqueeze(0), 
                torch.swapdims(self.W, 0, 1), 
                padding='valid')
        return WTX
    
    def get_dRdH(self, WTX, smooth_kernel):
        # Ensure 3D input to conv1d, then squeeze back to (K, T)
        WTXS = F.conv1d(
            WTX,  # shape (1, K, T)
            torch.tile(smooth_kernel, (self.K, 1)).unsqueeze(1),  # (K,1,2L-1)
            padding='same',
            groups=self.K
        ).squeeze(0)  # -> (K, T)

        if self.K == 1:
            # No off-diagonal terms exist when K=1
            dRdH = torch.zeros_like(WTXS)
        else:
            I = torch.eye(self.K, device=self.device, dtype=self.dtype)
            dRdH = (1 - I) @ WTXS  # (K,K) @ (K,T) -> (K,T)
        return self.lam * dRdH

    def get_dHHdH(self, smooth_kernel, iH):
        # Ensure 3D input to conv1d, then squeeze back to (K, T)
        HS = F.conv1d(
            self.H[iH].unsqueeze(0),  # (1, K, T)
            torch.tile(smooth_kernel, (self.K, 1)).unsqueeze(1),  # (K,1,2L-1)
            padding='same',
            groups=self.K
        ).squeeze(0)  # -> (K, T)

        if self.K == 1:
            # Off-diagonal ("i≠j") term is identically zero when only one component exists
            HSH_inej = torch.zeros_like(HS)
        else:
            I = torch.eye(self.K, device=self.device, dtype=self.dtype)
            HSH_inej = (1 - I) @ HS  # (K,K) @ (K,T) -> (K,T)

        return self.lambda_OrthH * HSH_inej
    
    def get_Wflat(self):
        return self.W.sum(-1)
    
    # def renormalize(self, iH):
    #     eps = torch.finfo(self.dtype).eps
    #     norms = torch.norm(self.H[iH], p=2, dim=1) + eps      # (K,)
    #     self.H[iH].data = self.H[iH].data / norms.unsqueeze(1)
    #     self.W.data *= norms[None, :, None]

    @torch.no_grad()
    def renorm_epoch(self):
        eps = torch.finfo(self.W.dtype).eps
        # per-factor mass, invariant-ish to L
        g = self.W.sum(dim=2).mean(dim=0)             # (K,)
        g = torch.clamp(g, min=eps)

        # reparameterize so WH stays the same
        self.W.data = self.W.data / g[None, :, None]  # W[:,k,:] /= g_k
        for s in range(self.nH):
            self.H[s].data = self.H[s].data * g[:, None]  # H_s[k,t] *= g_k

    @torch.no_grad()
    def renorm_epoch_based_on_W_norm(self):
        eps = torch.finfo(self.W.dtype).eps
        norm = torch.norm(self.W, dim=(0,2)) + eps  # (K,)
        self.W.data = self.W.data / norm[None, :, None]  # W[:,k,:] /= ||W_k||
        for s in range(self.nH):
            self.H[s].data = self.H[s].data * norm[:, None]  # H_s[k,t] *= ||W_k||

    @torch.no_grad()
    def renorm_epoch_based_on_H_norm(self):
        # concatenate all H's along time axis and compute norm
        eps = torch.finfo(self.W.dtype).eps
        H_concat = torch.cat(self.H, dim=1)  # (K, sum_t T_s)
        norm = torch.norm(H_concat, dim=1) + eps  # (K,)
        self.W.data = self.W.data * norm[None, :, None]  # W[:,k,:] *= ||H_k||
        for s in range(self.nH):
            self.H[s].data = self.H[s].data / norm[:, None]  # H_s[k,t] /= ||H_k||
    
    def get_XS(self, X, smooth_kernel):
        XS = F.conv1d(
            X,
            torch.tile(smooth_kernel, (self.N, 1)).unsqueeze(1), 
            padding='same', groups=self.N).squeeze()
        return XS
    
    def finalize(self, return_xhat=True):
        if return_xhat:
            Xhat_list = list()
            for iH in range(self.nH):
                Xhat_list.append(self.forward(iH).squeeze().to('cpu').detach().numpy()[:, self.L:-self.L])
                self.H[iH].data = self.H[iH].data[..., self.L:-self.L]

            return Xhat_list, self.W.data.to('cpu').detach().numpy(), [self.H[iH].data.to('cpu').detach().numpy() for iH in range(self.nH)]
        for iH in range(self.nH):
            self.H[iH].data = self.H[iH].data[..., self.L:-self.L]
        return self.W.data.to('cpu').detach().numpy(), [self.H[iH].data.to('cpu').detach().numpy() for iH in range(self.nH)]


    def shift_factors(self):
        W = self.W
        H = self.H

        Wpad = F.pad(W, (self.L, self.L))
        if self.L > 1:
            center = self.L // 2
            Wmass = W.sum(0)
            com = torch.floor( (torch.arange(1, self.L+1, device=self.device) * Wmass ).sum(1) / Wmass.sum(1) )
            bad_idx = torch.isnan(com)
            com = com.type(torch.IntTensor)
            for idx in torch.where(~bad_idx)[0]:
                Wpad[:,idx,:] = torch.roll(Wpad[:,idx,:], (center - com[idx].item(),), dims=1)
                for iH in range(self.nH):
                    H[iH][idx,:] = torch.roll(H[iH][idx,:], (com[idx].item() - center,), dims=0)
                    self.H[iH].data = H[iH]

        self.W.data = Wpad[:,:,self.L:-self.L]
        # self.H.data = H

    def _trim_for_eval(self, Z):
        # match forward()'s left-pad; use symmetric (L-1) trimming
        if self.L <= 1: 
            return Z
        return Z[..., (self.L-1):-(self.L-1)]  # not self.L:-self.L

    def compute_cost(self, X, iH, Mt):
        Xhat_full = self.forward(iH)[0]            # (N, T)
        X_full    = X[0]                            # (N, T)
        Mt_full   = Mt[0,0]                         # (T,)

        Xhat = self._trim_for_eval(Xhat_full)
        Xtrm = self._trim_for_eval(X_full)
        Mtt  = self._trim_for_eval(Mt_full).cpu().numpy()

        Xhat = Xhat.detach().cpu().numpy()[:, ~Mtt]
        Xtrm = Xtrm.detach().cpu().numpy()[:, ~Mtt]

        resid = Xtrm - Xhat
        cost = np.sqrt((resid**2).sum())
        r2 = np.mean(1 - (resid**2).sum(-1) / ((Xtrm - Xtrm.mean(-1, keepdims=True))**2 + 1e-8).sum(-1))
        return cost, r2

    def compute_xortho(self, X, iH):
        Xhat = self.forward(iH)[..., self.L:-self.L]
        WTX = self.get_WTX(X.squeeze())
        smooth_kernel = torch.ones(1, 2*self.L - 1, dtype=self.dtype).to(self.device)
        WTXS = F.conv1d(
            WTX,
            torch.tile(smooth_kernel, (self.K, 1)).unsqueeze(1),
            padding='same', groups=self.K).squeeze(0)
        WTXSHT = (WTXS @ self.H[iH].T) @ ( 1 - torch.eye(self.K, device=self.device, dtype=self.dtype) )
        reg = torch.linalg.norm(WTXSHT)
        return reg

    def do_mult_update_step(self, X, iH, M=None):
        smallnum = X.max()*1e-6
        eps = torch.finfo(X.dtype).eps
        
        smooth_kernel = torch.ones(1, 2*self.L - 1, dtype=self.dtype).to(self.device)
        X_hat = self.forward(iH)
        X_eff = X.clone()
        if M is not None:
            X_eff[M] = X_hat[M]   # ignore masked entries for H

        WTX = self.get_WTX(X_eff.squeeze())
        WTXhat = self.get_WTX(X_hat.squeeze())

        if self.lam > 0:
            dRdH = self.get_dRdH(WTX, smooth_kernel)
            
        else:
            dRdH = 0
            
        if self.lambda_OrthH > 0:
            dHHdH = self.get_dHHdH(smooth_kernel, iH)

        else:
            dHHdH = 0
        
        dRdH = dRdH + self.lambda_L1H + dHHdH
        
        # update H
        self.H[iH] *= (WTX / (WTXhat + dRdH + eps)).squeeze(0)
        
        # Shift to center factors
        if self.shift:
            self.shift_factors()
        self.W += smallnum
        
        # Renormalize so rows of H have constant energy
        # TODO: figure out what this is
        # self.renormalize(iH)
        
        # The updated implementation so the mask works.
        W_num, W_den = None, None
        if not self.W_fixed:
            Xhat = self.forward(iH)
            if M is not None:
                X[M] = Xhat[M]

            if self.lambda_OrthW > 0:
                Wflat = self.W.sum(-1)
            
            if (self.lam > 0) & self.use_W_update:
                XS = self.get_XS(X, smooth_kernel)
            
            W_num = torch.zeros(self.W.shape, device=self.W.device)
            W_den = torch.zeros(self.W.shape, device=self.W.device)
            for l in range(self.L):
                H_shifted = torch.roll(self.H[iH], l, dims=1)
                # print(l, self.H.shape, X.shape, H_shifted.shape, Xhat.shape)
                XHT = X @ H_shifted.T
                XhatHT = Xhat @ H_shifted.T
                
                if self.lam > 0 and self.use_W_update:
                    dRdW = self.lam * XS @ H_shifted.T @ ( 1 - torch.eye(self.K, device=self.device, dtype=self.dtype) )
                else:
                    dRdW = 0
                    
                if self.lambda_OrthW > 0:
                    dWWdW = self.lambda_OrthW * Wflat @ ( 1 - torch.eye(self.K, device=self.device, dtype=self.dtype) )
                else:
                    dWWdW = 0
                    
                dRdW += self.lambda_L1W + dWWdW
                
                W_num[..., l] = XHT
                W_den[..., l] = (XhatHT + dRdW).squeeze(0)
        else:
            return None, None

        return W_num, W_den
