"""Lithium inventory and equilibrium initialization for the SPMeT pulse replay."""

from __future__ import annotations

import math

F = 96485.33212331001


def derive_updates(p,s_n,s_p,delta):
 if not all(math.isfinite(float(x)) for x in (s_n,s_p,delta)) or not .85<=s_n<=1 or not .85<=s_p<=1 or not -.4<=delta<=.4:raise ValueError('Unchanged bounded electrode/inventory domain required')
 A=p['Electrode width [m]']*p['Electrode height [m]'];a={};coeff={}
 for side,scale in [('Negative',s_n),('Positive',s_p)]:
  active=p[side+' electrode active material volume fraction']*scale
  if not 0<active<=1-p[side+' electrode porosity']+1e-12:raise ValueError('Active plus pore volume invalid')
  a[side+' electrode active material volume fraction']=active
  coeff[side]=F*A*p[side+' electrode thickness [m]']*active/3600
 cn=p['Initial concentration in negative electrode [mol.m-3]']+delta/coeff['Negative'];cp=p['Initial concentration in positive electrode [mol.m-3]']
 if not 0<cn<p['Maximum concentration in negative electrode [mol.m-3]']:raise ValueError('Inventory shift outside physical stoichiometry')
 a['Initial concentration in negative electrode [mol.m-3]']=cn
 return a,dict(Q_n_Ah=coeff['Negative']*p['Maximum concentration in negative electrode [mol.m-3]'],Q_p_Ah=coeff['Positive']*p['Maximum concentration in positive electrode [mol.m-3]'],Q_Li_Ah=coeff['Negative']*cn+coeff['Positive']*cp)


def homogeneous_state(p,inventory,target):
 from scipy.optimize import brentq
 Qn,Qp,Li=[inventory[k] for k in ('Q_n_Ah','Q_p_Ah','Q_Li_Ah')]
 lo=max(1e-6,(Li-Qp*(1-1e-6))/Qn);hi=min(1-1e-6,(Li-Qp*1e-6)/Qn)
 if hi<=lo:raise ValueError('No feasible lithium-conserving stoichiometry interval')
 def voltage(x):
  y=(Li-Qn*x)/Qp;a=p['Positive electrode OCP [V]'](y)-p['Negative electrode OCP [V]'](x)
  import numpy as np
  return float(np.asarray(a.evaluate() if hasattr(a,'evaluate') else a).item())
 def at(v):return float(brentq(lambda x:voltage(x)-v,lo,hi,xtol=1e-10,maxiter=50))
 x=at(target);y=(Li-Qn*x)/Qp;x0=at(2.5);x100=at(4.2)
 if not x0<x<x100:raise ValueError('Initial rest state outside reviewed voltage window')
 if abs(Qn*x+Qp*y-Li)>1e-10:raise ValueError('Initial lithium inventory was not conserved')
 return {'Initial concentration in negative electrode [mol.m-3]':x*p['Maximum concentration in negative electrode [mol.m-3]'],'Initial concentration in positive electrode [mol.m-3]':y*p['Maximum concentration in positive electrode [mol.m-3]']},dict(negative_stoichiometry=x,positive_stoichiometry=y,equilibrium_SOC=(x-x0)/(x100-x0),initial_equilibrium_V=voltage(x),target_rest_endpoint_V=target,lithium_inventory_residual_Ah=Qn*x+Qp*y-Li)
