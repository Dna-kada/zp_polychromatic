# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 04:58:48 2024

@author: Dunka
"""
from scipy.special import j0
from time import process_time
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from parameters import *




def run_tasks(width_rank):
    peaks = []
    crosses = []
    q=0
    for a in width_rank:
        width_cross = []
        width_peak = []
        for j in foc:
            FWHMl = lamda/a
            dlamda = FWHMl/2.3548  # (2*np.sqrt(np.log(2)))
            length = 13
            wav = np.linspace(lamda-2*dlamda,lamda+2*dlamda,length,dtype=np.complex128)
            Fim1 = np.zeros(N//2)
            poly = Gauss_spec(wav,lamda,dlamda)*np.exp(0j)
            for i in range(len(poly)):
                tt1 = process_time()
                Fim1 += abs(poly[i]*iHT(te/scale,prop(te,HT_FZP,j,wav[i],dr,kr),R,dr,kr))**2
                tt2 = process_time()
                # print(tt2-tt1)
            IP = Fim1
            width_cross.append(Fim1)
            width_peak.append(np.max(IP))
            q += 1
            # print(f"q= {q}")
        crosses.append(width_cross)
        peaks.append(width_peak)
    return np.array(crosses),np.array(peaks)



def run_tasks_mono():
    peaks = []
    crosses = []
    q=0
    for j in foc:
        Fim1 = abs(iHT(te/scale,prop(te,HT_FZP,j,lamda,dr,kr),R,dr,kr))**2
        crosses.append(Fim1)
        peaks.append(np.max(Fim1))
        q += 1
    return np.array(crosses),np.array(peaks)


# ----- Main -----
#%%

FZP,rr,fz = ZP_cD_ph(te,nz,R,lamda)
#%%
# plt.figure()
# plt.plot(te,FZP,'.')
#%%
R = rr[-1]
delr = rr[-1] - rr[-2]
print(f"delr : {delr}")
print(f"R = {R}")
HT_FZP = HT(te,FZP,R,dr,kr)
first_null = 2*1.22*R/(4*(nz))
NA = np.sin(np.arctan((R)/fz))
f_l = fz - 4*(0.5*lamda/(NA**2))
f_f = fz + 4*(0.5*lamda/(NA**2))
fl = 15

foc = np.linspace(f_l,f_f,fl)

ims = []
scale = 10
q = 0
crosses = []
peaks = []


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

f = sort(size,wl)
s = int(f[:rank].sum())
e = int(s + f[rank])
print([s,e])
t1 = process_time()
rank_crosses,rank_peaks = run_tasks(width[s:e])
print(f"Rank {rank} - processed")
t2 = process_time()
print(f"Time taken: {t2-t1}")

if rank != 0:
    comm.send([rank_crosses,rank_peaks], dest=0)
    print(f"{rank} - sent")



elif rank == 0:
    crosses = rank_crosses
    peaks = rank_peaks
    
    for l in range(1,size):
        res = comm.recv(source=l)
        orc = res[0]
        orp = res[1]
        print(f"{l} - received")
        peaks = np.concatenate((peaks,orp))
        crosses = np.concatenate((crosses,orc))
    print("all done")

if rank == 0:
    # print(peaks)
    print(np.shape(np.array(peaks)))
    name = str(input("Enter tag:"))
    for i in range(wl):
        np.savetxt(f"peaks_{name}_{i}.data", np.array(peaks[i]))
        np.savetxt(f"crosses_{name}_{i}.data", np.array(crosses[i]))
    c_mono,peak_mono = run_tasks_mono()
    print(peak_mono)
    np.savetxt(f"crosses_{name}_mono.data",c_mono)
    np.savetxt(f"peaks_{name}_mono.data",peak_mono)


#
# for i in range(wl):
#     np.savetxt("crosses_DOF_poly_{i].txt",crosses[i]])
#     np.savetxt("peaks_DOF_poly_{i].txt",peaks[i]])