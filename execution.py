# -*- coding: utf-8 -*-
"""Parallel execution of zone plate simulations."""
import numpy as np
from time import process_time
from mpi4py import MPI
from parameters import *


def run_tasks(width_rank):
    """Process a slice of the width array on a single MPI rank."""
    peaks = []
    crosses = []
    q = 0
    for a in width_rank:
        width_cross = []
        width_peak = []
        for j in foc:
            FWHMl = lamda / a
            dlamda = FWHMl / 2.3548  # (2*np.sqrt(np.log(2)))
            length = 13
            wav = np.linspace(
                lamda - 2 * dlamda, lamda + 2 * dlamda, length, dtype=np.complex128
            )
            Fim1 = np.zeros(N // 2)
            poly = Gauss_spec(wav, lamda, dlamda) * np.exp(0j)
            for i in range(len(poly)):
                tt1 = process_time()
                Fim1 += (
                    abs(
                        poly[i]
                        * iHT(
                            te / scale, prop(te, HT_FZP, j, wav[i], dr, kr), R, dr, kr
                        )
                    )
                    ** 2
                )
                tt2 = process_time()
                # print(tt2-tt1)
            IP = Fim1
            width_cross.append(Fim1)
            width_peak.append(np.max(IP))
            q += 1
            # print(f"q= {q}")
        crosses.append(width_cross)
        peaks.append(width_peak)
    return np.array(crosses), np.array(peaks)


def run_tasks_mono():
    """Compute intensity for a monochromatic source."""
    peaks = []
    crosses = []
    q = 0
    for j in foc:
        Fim1 = abs(iHT(te / scale, prop(te, HT_FZP, j, lamda, dr, kr), R, dr, kr)) ** 2
        crosses.append(Fim1)
        peaks.append(np.max(Fim1))
        q += 1
    return np.array(crosses), np.array(peaks)


def main():
    """Entry point for running the MPI simulation."""
    FZP, rr, fz = ZP_cD_ph(te, nz, R, lamda)
    R = rr[-1]
    delr = rr[-1] - rr[-2]
    print(f"delr : {delr}")
    print(f"R = {R}")
    HT_FZP = HT(te, FZP, R, dr, kr)
    NA = np.sin(np.arctan((R) / fz))
    f_l = fz - 4 * (0.5 * lamda / (NA**2))
    f_f = fz + 4 * (0.5 * lamda / (NA**2))
    fl = 15

    global foc, scale
    foc = np.linspace(f_l, f_f, fl)
    scale = 10

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    f = sort(size, wl)
    s = int(f[:rank].sum())
    e = int(s + f[rank])
    t1 = process_time()
    rank_crosses, rank_peaks = run_tasks(width[s:e])
    print(f"Rank {rank} - processed")
    t2 = process_time()
    print(f"Time taken: {t2 - t1}")

    if rank != 0:
        comm.send([rank_crosses, rank_peaks], dest=0)
        print(f"{rank} - sent")
        return

    crosses = rank_crosses
    peaks = rank_peaks

    for l in range(1, size):
        orc, orp = comm.recv(source=l)
        print(f"{l} - received")
        peaks = np.concatenate((peaks, orp))
        crosses = np.concatenate((crosses, orc))
    print("all done")

    print(np.shape(np.array(peaks)))
    name = str(input("Enter tag:"))
    for i in range(wl):
        np.savetxt(f"peaks_{name}_{i}.data", np.array(peaks[i]))
        np.savetxt(f"crosses_{name}_{i}.data", np.array(crosses[i]))
    c_mono, peak_mono = run_tasks_mono()
    print(peak_mono)
    np.savetxt(f"crosses_{name}_mono.data", c_mono)
    np.savetxt(f"peaks_{name}_mono.data", peak_mono)


if __name__ == "__main__":
    main()
