import sys; sys.path.insert(0,"/Users/yanlu/Documents/glowing-journey")
import torch, pyro, numpy as np, math, time
import c302
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro_svi_c302 import init_to_prior_median
pyro.clear_param_store(); pyro.set_rng_seed(0); torch.manual_seed(0)
syn={"General":([], [], []), "Wicks":([0],[1],[1.0],[0.0],[0.01]), "Generic":([], [], [])}
step=c302.NematodeForStep(3, {}, syn)
rec=c302.RecurrentNematode(step,dt_obs=0.5,substeps=2)
B,T,N=4,20,3
stim=torch.zeros(B,T,N); stim[:,5,0]=1.0
true={
 "model.Neuron.conductance.E":torch.full((N,),-0.065),
 "model.Neuron.conductance.G":torch.full((N,),0.5),
 "model.Neuron.conductance.C":torch.full((N,),0.05),
 "model.synapse.wicks.g_max":torch.tensor([0.05]),
 "model.synapse.wicks.V_rest":torch.full((N,),-0.042),
 "model.synapse.wicks.V_slope":torch.full((N,),0.015),
 "model.synapse.wicks.E_rev":torch.tensor([0.0]),
 "model.stim_gain":torch.tensor(1.2),
 "obs_sigma":torch.tensor(0.15),
 "gcamp.v_th_ca_logit":c302.CalciumGCaMPLayer.v_th_ca_to_logit(-0.040),
 "gcamp.v_slope_ca":torch.tensor(0.008),
 "gcamp.alpha_ca":torch.tensor(2.5),
 "gcamp.tau_ca":torch.tensor(7.0),
 "gcamp.k_on":torch.tensor(0.06),
 "gcamp.k_off":torch.tensor(0.045),
 "gcamp.gain":torch.full((N,),1.4),
 "gcamp.offset":torch.zeros(N),
}
with torch.no_grad():
    with pyro.poutine.condition(data=true):
        y,_,_=rec.simulate(stim)
y=y+0.01*torch.randn_like(y)
mask=torch.ones_like(y,dtype=torch.bool)
noise=torch.full((B,N),0.05)
guide=AutoDiagonalNormal(rec, init_loc_fn=init_to_prior_median, init_scale=0.02)
svi=SVI(rec,guide,pyro.optim.ClippedAdam({'lr':0.01}),Trace_ELBO())
t0=time.time()
for i in range(300):
    loss=svi.step(ExternalInput=stim,mask=mask,y=y,noise_scale=noise)
med=guide.median()
print('time',round(time.time()-t0,1),'loss',loss)
for k in ['model.synapse.wicks.g_max','gcamp.tau_ca','gcamp.v_th_ca_logit','gcamp.k_off','model.Neuron.conductance.G']:
    v=med.get(k); print(k, None if v is None else (v.detach().cpu().tolist() if v.numel()<5 else float(v.mean())))
print('true gmax .05 tau 7 vthlogit',float(c302.CalciumGCaMPLayer.v_th_ca_to_logit(-0.040)),'koff .045 G .5')
