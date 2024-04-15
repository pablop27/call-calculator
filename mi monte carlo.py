# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:54:38 2024

@author: pablo
"""

import numpy as np
import random
from tqdm import tqdm
def prec_opc(simul,T,cam_dia,strike,p_hoy,sigma,tasa,max_koo,bn_thresh,bn_pay):
    eur_call=[]
    asn_call=[]
    look_back_opt=[]
    knock_out_opt=[]
    binary_opt=[]
    asn_binary_opt=[]
    look_back_binary_opt=[]
    dias=round(T*252) # cuántos días van a ser simulados
    t=dias*cam_dia # cuantos precios distintos van a ser simulados
    dt=T/t 
    hp=np.zeros([simul,t+1]) # guardar las simulaciones
    for g in tqdm(range(simul),colour="red"):
        prec=np.zeros([t+1,1]) # matriz dónde vana  estar los precios de la actual simulación, incluyendo el precio inicial
        prec[0,0]=p_hoy
        w=np.random.normal(loc=0, scale=1, size=(t,1))*np.sqrt(dt) # generar la parte estocástica para los t momentos a simular
        alf=tasa*dt*np.ones([t,1]) # parte determinística de los t momentos a simular
        aum=sigma*w+alf+np.ones([t,1]) # factor de aumento para cada periodo
        prec[1:,0]=p_hoy*np.cumprod(aum) # usar la productoria para calcular el precio en cada t
        ec=np.max([0,prec[-1,0]-strike]) # calcular el valor de la opción dado el último precio
        ac=np.max([0,np.mean(prec[:,0])-strike])
        lbo=np.max([0,np.max(prec[:,0])-strike])
        pkoo=prec[-1,0]
        if pkoo>max_koo:
            pkoo=0
        koo=np.max([0,pkoo-strike])
        pbo=prec[-1,0]
        if pbo>bn_thresh:
            bo=bn_pay
        else:
            bo=0
        pabo=np.mean(prec[:,0])
        if pabo>bn_thresh:
            abo=bn_pay
        else:
            abo=0
        plbbo=np.max(prec[:,0])
        if plbbo>bn_thresh:
            lbbo=bn_pay
        else:
            lbbo=0
        # guardar los valores de las distintas  opciones de todas las simulaciones
        eur_call.append(ec) 
        asn_call.append(ac)
        look_back_opt.append(lbo)
        knock_out_opt.append(koo)
        binary_opt.append(bo)
        asn_binary_opt.append(abo)
        look_back_binary_opt.append(lbbo)
        # se guardan todas las simulaciones en hp por si se quiere evaluar una opción más exótica.
        hp[g,:]=prec.reshape(t+1)
    print(f"Los precios de las opciones con un strike price de {strike} y que vencen en {T} años son: ")
    print(f"European call: {np.mean(np.array(eur_call))*np.exp(-tasa*T)}")
    print(f"Asian call: {np.mean(np.array(asn_call))*np.exp(-tasa*T)}")
    print(f"Look back option: {np.mean(np.array(look_back_opt))*np.exp(-tasa*T)}")
    print(f"Knock out option con límite de {max_koo}: {np.mean(np.array(knock_out_opt))*np.exp(-tasa*T)}")
    print(f"En el caso de las binarias, cuando el límite es {bn_thresh} y el pago es {bn_pay}:")
    print(f"Binary: {np.mean(np.array(binary_opt))*np.exp(-tasa*T)}")
    print(f"Asian binary: {np.mean(np.array(asn_binary_opt))*np.exp(-tasa*T)}")
    print(f"Look back binary: {np.mean(np.array(look_back_binary_opt))*np.exp(-tasa*T)}")
    return hp

def prec_opcr(simul,T,cam_dia,strike,p_hoy,sigma,tasa,max_koo,bn_thresh,bn_pay): # forma más rápida de llegar a los resultados
    # si bien en teoría es una forma más rápida, consume demasiada ram y para muchas simulaciones no es viable
    dias=round(T*252) # cuántos días van a ser simulados
    t=dias*cam_dia # cuantos precios distintos van a ser simulados
    dt=T/t 
    for pl in tqdm(range(1),colour="red"):
        w=np.random.normal(loc=0, scale=1, size=(simul,t+1))*np.sqrt(dt) # generar la parte estocástica para los t momentos a simular
        w[:,0]=0
        alf=tasa*dt*np.ones([simul,t+1]) # parte determinística de los t momentos a simular
        alf[:,0]=0
        aum=sigma*w+alf+np.ones([simul,t+1]) # factor de aumento para cada periodo
        prec=p_hoy*np.cumprod(aum,axis=1) # usar la productoria para calcular el precio en cada t
        # en prec quedaron todas las simualciones
        ec=np.max(np.hstack((np.zeros([simul,1]),(prec[:,-1]-strike).reshape(simul,1))),axis=1) 
        ac=np.max(np.hstack((np.zeros([simul,1]),(np.mean(prec[:,:],axis=1)-strike).reshape(simul,1))),axis=1)
        lbo=np.max(np.hstack((np.zeros([simul,1]),(np.max(prec[:,:],axis=1)-strike).reshape(simul,1))),axis=1) 
        pkoo=prec[:,-1].copy()
        pkoo[pkoo>max_koo]=0
        koo=np.max(np.hstack((np.zeros([simul,1]),(pkoo-strike).reshape(simul,1))),axis=1)
        pbo=prec[:,-1].copy()
        bo=np.zeros([simul,1])
        bo[pbo>bn_thresh,0]=bn_pay
        pabo=np.mean(prec[:,:],axis=1)
        abo=np.zeros([simul,1])
        abo[pabo>bn_thresh,0]=bn_pay
        plbbo=np.max(prec[:,:],axis=1)
        lbbo=np.zeros([simul,1])
        lbbo[plbbo>bn_thresh,0]=bn_pay
        # guardar los valores de las distintas  opciones de todas las simulaciones
    print(f"Los precios de las opciones con un strike price de {strike} y que vencen en {T} años son: ")
    print(f"European call: {np.mean(ec)*np.exp(-tasa*T)}")
    print(f"Asian call: {np.mean(ac)*np.exp(-tasa*T)}")
    print(f"Look back option: {np.mean(lbo)*np.exp(-tasa*T)}")
    print(f"Knock out option con límite de {max_koo}: {np.mean(koo)*np.exp(-tasa*T)}")
    print(f"En el caso de las binarias, cuando el límite es {bn_thresh} y el pago es {bn_pay}:")
    print(f"Binary: {np.mean(bo)*np.exp(-tasa*T)}")
    print(f"Asian binary: {np.mean(abo)*np.exp(-tasa*T)}")
    print(f"Look back binary: {np.mean(lbbo)*np.exp(-tasa*T)}")
    return prec

#%%
#simulsr=prec_opcr(900000,1/12,50,40,40,0.1,0.0517,40.5,40,10)
simuls=prec_opc(900000,1/12,50,40,40,0.1,0.0517,40.5,40,10)


