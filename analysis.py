# -*- coding: utf-8 -*-
"""Post-processing utilities for analyzing simulation results."""
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import fft, fft2, ifft2, fftfreq, fftshift, ifftshift
from scipy.special import j0
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d, UnivariateSpline, CubicSpline, UnivariateSpline
from scipy.optimize import curve_fit
from parameters import *

# %%
mpl.rcParams["figure.figsize"] = (5, 3.5)
mpl.rcParams["axes.titlesize"] = 15
mpl.rcParams["axes.labelsize"] = 15
mpl.rcParams["legend.fontsize"] = 12
fac = int(2)


mm = 1e-3
nm = 1e-9
um = 1e-6


# ---- Simulation -----
Dia = 126e-6
R = Dia / 2
lamda = 2.73 * nm
nz = 900
N = nz * 25 // fac

scale = 10
L2 = R / 0.30 / scale
L = R / 0.30 / scale
dx = L / N
dr = dx
x = np.linspace(-L2, L2, N * 2)
fx = np.arange(-1 / dx, 1 / dx, 1 / (L2))
X, Y = np.meshgrid(x, x)
fz = R**2 / (nz * lamda)  # 0.0016147703653846153
f1 = np.ones(np.shape(X))
NA = np.sin(np.arctan((R) / fz))
te = np.linspace(0, L, N)
fr = np.linspace(0, 1 / (2 * dr), N)
kr = 2 * np.pi * fr
delr = 3.50250686146058e-08
delr = lamda / (2 * NA)

fl_mono = fz - (0.5 * lamda / (NA**2))
ff_mono = fz + (0.5 * lamda / (NA**2))
f_l = fz - 4 * (0.5 * lamda / (NA**2))
f_f = fz + 4 * (0.5 * lamda / (NA**2))

foc = np.linspace(f_l, f_f, 15)
focl = len(foc)
w_0 = 0.05
w_l = 1.1
width = np.linspace(nz * w_0, nz * w_l, 18)
wl = len(width)
first_null = 2 * 1.22 * R / (4 * (nz))
delr = Dia / (4 * (nz))
cut_ = 1 / (1.22 * delr)

cross = []
peaks = []
for i in range(wl):
    cross.append(np.loadtxt(f"crosses_updated_v5_{i}.data"))
    peaks.append(np.loadtxt(f"peaks_updated_v5_{i}.data"))

cm = np.loadtxt("crosses_updated_v5_mono.data")
pm = np.loadtxt("peaks_updated_v5_mono.data")
# %%
vals_mono_full = []
# lines = ['k-','k-.','k--','k:']
plt.figure()
plt.title("MTF for different ζ in focus")
plt.ylabel("Modulation")
plt.xlabel("frequency (cyc/μm)")
# plt.xlim(0,cut_*(1e-6)) # 3.32e8
zzz = 0
for q in [0.05, 0.1, 0.15]:
    vals_mono = []
    for i in range(len(foc)):
        fftt = abs(HT(te, cm[i], R, dr, kr))
        ft = fftt / np.max(fftt)
        f = interp1d(fr, ft, kind="cubic")
        ri = np.linspace(fr[0], 1.5 / (first_null), 1000)
        yi = f(ri)
        plt.plot(ri, yi)
        # plt.xlim(0,1.1*(NA/lamda))
        plt.xlim(0, 1 / (lamda * fz / Dia))
        plt.legend(np.round(width, decimals=0))
        zzz += 1
        if i == 0:
            monoo_oof = yi
        if i == len(foc) // 2:
            monoo = yi
        for a in range(len(yi)):
            if yi[a] < q:
                vals_mono.append((ri[a] + ri[a - 1]) / 2)
                break
    vals_mono_full.append(vals_mono)

crosses = np.array(cross)

# %%
mpl.rcParams["figure.figsize"] = (5, 3.5)
plt.figure()
plt.title(f"MTF curves at ± {round(fz*1e6-f_l*1e6,1)} mm")
plt.ylabel("Modulation")
plt.xlabel("cyc/m")
v = [0, 2, 17]
l = ["k-", "k:", "k--", "k-."]
plt.xlim(0, 1.5 / (first_null))
plt.plot(ri, monoo_oof, l[0])
for i in range(1, 4):
    fftt = abs(HT(te, crosses[v[i - 1]][0], R, dr, kr))
    ft = fftt / np.max(fftt)
    plt.plot(fr, ft, l[i])
plt.legend(["monochromatic", *np.round(width[v] / nz, 2)])
plt.tight_layout()
plt.savefig("oof_MTF.png")

# %%
vals_full = []
plt.figure()
MTFs = []
plots = []
for q in [0.05, 0.1, 0.15]:
    vals = []
    for i in range(wl):  # [ 1+ len(foc)//2]
        plt.title("MTF for different ζ in focus")
        plt.ylabel("Modulation")
        plt.xlabel("frequency (cyc/μm)")
        # plt.xlim(0,cut_*(1e-6)) # 3.32e8
        zzz = 0
        for j in [focl // 2]:
            fftt = abs(HT(te, crosses[i][focl // 2], R, dr, kr))
            ft = fftt / np.max(fftt)
            f = interp1d(fr, ft, kind="linear")
            ri = np.linspace(fr[0], 1.5 / (first_null), 1000)
            yi = f(ri)
            plt.plot(ri, yi)
            plt.xlim(0, 1 / (lamda * fz / Dia))
            plt.legend(np.round(width, decimals=0))
            if q == 0.05:
                MTFs.append(yi)
                plots.append(yi)
            zzz += 1
            for a in range(len(yi)):
                if yi[a] < q:
                    vals.append((ri[a] + ri[a - 1]) / 2)
                    break
    vals_full.append(vals)

plt.tight_layout()
plt.grid()

mpl.rcParams["figure.figsize"] = (5, 3.5)
plt.figure()
plt.title("MTF for different ζ in focus")
plt.ylabel("Modulation")
plt.xlabel("frequency (cyc/μm)")
v = [0, 2, 9, 17]
l = ["k-", "k:", "k", "k--", "k-."]
plt.plot(ri, monoo, l[0])
for i in range(1, 5):
    plt.plot(ri, MTFs[v[i - 1]], l[i])
plt.legend(["monochroatic", *np.round(width[v] / nz, 2)])
plt.tight_layout()
plt.savefig("MTF_in_focus.png")
# %%
l = ["kd-", "k-.", "k--"]
plt.figure()
for v in range(len(vals_full)):
    plt.plot(width / nz, vals_full[v] / np.max(vals_mono_full[v]), l[v], markersize=5)
plt.title("Fraction of monochr. value ")
plt.xlabel("ζ/Nz")
plt.ylabel("fraction")
plt.ylim(0.15, 1.1)
plt.grid()
plt.legend(["5%", "10%", "15%"])
plt.tight_layout()
plt.savefig("const_r_demonstration.png")

vals = vals_full[1]
vals_mono = vals_mono_full[1]
max_v = np.max(vals_mono)
fit_res = CubicSpline(width / nz, vals / max_v)

monochr = 300
NZ = np.linspace(monochr * 0.9, monochr * 7, 400)
NZdw = NZ / monochr

n_w = nz / width
f_cut_10p = interp1d(width / nz, (vals / max_v))  # , kind="cubic"
n_d_width = np.linspace(n_w[-1], n_w[2], 400)
mono_curve = 1 / (1.22 * Dia / (4 * (n_d_width)))
plt.figure()
plt.title("constant zp")
plt.plot(width / nz, (vals / max_v))
fn = f_cut_10p(1 / n_d_width)

plt.figure()
plt.title("decrease factor")
plt.plot(n_d_width, fn)

# This returns the optimal monochromaticity for the curve.
der = CubicSpline(n_d_width, fn * mono_curve)

# plt.figure()
der_data = der(n_d_width, nu=1)
plt.figure()
plt.title("derivative decrease")
plt.plot(n_d_width, der_data)
plt.figure()
plt.plot(der_data)
plt.plot(NZdw, der_data, "k-.")
plt.tight_layout()
opt_val = 0
opt_y = 0
for i in range(len(der_data)):
    if (der_data)[i] <= 0:
        opt_val = (NZdw)[i]
        opt_y = der(opt_val, nu=0)
# %%

mpl.rcParams["figure.figsize"] = (5, 3.5)

plt.figure()
plt.title("Rel. of Cut-off freq. to ")
plt.plot(n_d_width, mono_curve, "k--")
plt.plot(n_d_width, (fn) * mono_curve, "k-.")
# plt.plot(NZdw,cut_offs)
plt.xlabel("Nz/ζ")
plt.ylabel("cut-off freq [10%]")
plt.legend(["monochromatic", "polychromatic"])
# plt.plot(opt_val,opt_y,'x')
plt.tight_layout()
plt.grid()
np.savetxt("nzs_width_pred.txt", [n_d_width, (fn) * mono_curve])
# return number of zones that produce the equivalent resolution given
# monochromatic illumination.
line_fit = UnivariateSpline(fn * mono_curve, n_d_width, k=3)
eq_NZdw = line_fit(opt_y)
eq_NZ = eq_NZdw * monochr
print(
    "The equivalent number of zones that would achieve the same",
    f"resolution under monochromatic conditions equals {int(eq_NZ)}",
)
# %%
plt.figure()
plt.plot(n_d_width, (fn) * mono_curve)
plt.plot(opt_val, line_fit(opt_y), "x")
# %%
plt.figure()
for i in range(len(width)):
    plt.title("PSF for different monochromaticity out of focus")
    plt.ylabel("intensity")
    plt.xlabel("x-axis (nm)")
    plt.xlim(0, 1 / (1 * (NA / lamda)))
    fftt = crosses[i][6]
    ft = fftt / np.max(fftt)
    plt.plot(te, ft)
plt.legend(np.round(width, decimals=1))

# %%
q = 3
plt.figure()
plt.title(f"MTF through 3 DOF for width{round(width[q]/nz,2)}*no_zones")
plt.ylabel("Modulation")
plt.xlabel("cyc/um")
# arr = np.zeros()

for j in [0, 7, 12]:
    fftt = abs(HT(te, crosses[q][j], R, dr, kr))
    ft = (fftt) / np.max(fftt)
    plt.xlim(0, 1 * um / (lamda * fz / Dia))
    plt.plot(fr * um, ft)
    plt.legend(np.round(foc * 1e6, decimals=1))
    print(i * focl + j)


# %%
plt.figure()
plt.title("plot of peak cross-sections")
plt.xlabel("monochromaticity")
plt.ylabel("Intensity of spot centre")
peaks_foc = peaks
DOF = []
foc_rec = np.linspace(foc[len(foc) // 2], foc[-1], 1000, endpoint=True)
foc_rec2 = np.linspace(foc[0], foc[len(foc) // 2], 1000, endpoint=True)
for i in np.arange(0, len(width), 1):
    plt.plot(foc * 1e6, (peaks_foc)[i] / np.max(peaks_foc[i]))
    f_p = CubicSpline(foc, peaks_foc[i])
    f_rec = f_p(foc_rec, nu=0)
    f_rec_2 = f_p(foc_rec2, nu=0)
    f_rec_2 = f_rec_2 / np.max(f_rec_2)
    f_rec = f_rec / np.max(f_rec)
    DOF1 = 0
    DOF2 = 0
    for k in range(len(f_rec)):
        if f_rec[k] < 0.8:
            DOF1 = (foc_rec[k - 1] + foc_rec[k]) / 2 - foc_rec[0]
            break
    for k in range(len(f_rec_2)):
        if np.flip(f_rec_2)[k] < 0.8:
            DOF2 = (
                np.flip(foc_rec2)[0]
                - (np.flip(foc_rec2)[k - 1] + np.flip(foc_rec2)[k]) / 2
            )
            break
    DOF.append(DOF1 + DOF2)
plt.grid()
plt.tight_layout()

plt.savefig("peak_cross_DOF.png")

# %%
DOFn = np.array(DOF)
mpl.rcParams["figure.figsize"] = (4, 2.6)
plt.figure()
DOF_mono = (ff_mono - fl_mono) * 1e6
# plt.title("DOF as a function of wpz")
plt.plot((width / nz)[1:], (DOFn)[1:] * 1e6 / DOF_mono, "k.")
# plt.plot([(width/nz)[0],(width/nz)[-1]],np.array([DOF_mono,DOF_mono])/DOF_mono,'k')
cc = np.linspace(0.2, (width / nz)[-1], 400)

fit = CubicSpline((width / nz)[1:], DOFn[1:] * 1e6 / DOF_mono)
dat = fit(cc, nu=0)
plt.plot(cc, dat, "k-.")
plt.xlabel("ζ/Nz")
plt.ylabel("DOF scale")
plt.grid()
plt.legend(["data", "fit"])
plt.tight_layout()
plt.ylim(0.9, 3.5)
plt.savefig("DOF_v_wpz_scale.png")

# DOF_opt = fit(1/opt_val)
# plt.plot(1/opt_val,DOF_opt,'rx')

DOF_fit = CubicSpline(width / nz, DOFn)
# %%
# plt.figure()
# plt.title("FREQ_both")
# plt.plot(NZdw,(f_cut_10p(NZdw)))
# plt.xlabel("(no. of zones)/monochromaticity")
# plt.ylabel("cut-off freq (cyc/m)")
# plt.tight_layout()
# # plt.legend(["analytic","simulated"])
# plt.grid()

# %%
fftt = HT(te, (cross[5][len(foc) // 2]), R, dr, kr)

plt.figure()
plt.plot(te, cross[5][len(foc) // 2])
ft = fftt / np.max(fftt)
plt.figure()
plt.plot(ft)
# for i in range(5):
#     ft[i] = ft[7]
plt.figure()
plt.plot(ft)
M = 1
frm = fr / M
f = interp1d(fr, ft)

# %%
# dr = 13.5e-6
r3 = np.linspace(-39591315, 39591315, 2048)


# plt.figure()
# plt.plot(r3,f(r3))
# plt.plot(fr,ft)
pixel_size = r3[1] - r3[0]

# %%
RX3, RY3 = np.meshgrid(r3, r3)
# plt.figure()
# plt.imshow(np.sqrt(RX3**2+RY3**2))
# %%
field = f(np.sqrt(RX3**2 + RY3**2))
block = np.sqrt(RX3**2 + RY3**2) < (np.max(r3) / 2)

# ----- making a 2D pattern from the psf ----
fxmm = np.max(fx)
# np.savetxt("PSF_900.txt",field)
# %%
plt.figure()
plt.contourf((field * block))

plt.figure()
plt.plot(r3, field[len(field) // 2])

np.savetxt(f"TF_{nz}.txt", field * block)

# ----- adjusting for smallest zone width
# %%


# Specific examples used for calculating values for specific zone plates
# or monochromaticity values
delr = Dia / (4 * (NZ))  # array

plt.figure()
plt.title("Rel. of Cut-off freq. to ")
plt.plot(delr / nm, (1 / (1.22 * delr)), "k-")
plt.plot(delr / nm, np.flip(f_cut_10p(n_d_width)) * (1 / (1.22 * delr)), "k-.")
# plt.plot(NZdw,cut_offs)
plt.xlabel("smallest zone width (nm)")
plt.ylabel("cut-off freq (cyc/m)")
plt.tight_layout()
plt.legend(["monochromatic", "polychromatic"])
plt.grid()
plt.savefig("delr_900.png")

# %%
plt.figure()
plt.plot(delr / nm, (4 * delr**2) / (lamda), "k")
plt.plot(delr / nm, (4 * delr**2) / (lamda) * np.flip(dat), "k-.")
# plt.plot(NZdw,cut_offs)
plt.xlabel("smallest zone width (nm)")
plt.ylabel("DOF (m)")
plt.tight_layout()
plt.legend(["monochromatic", "polychromatic"])
plt.grid()
plt.xlim(30, 80)
plt.ylim(0, 1e-5)
plt.savefig("szw.png")
# %%
plt.figure()
plt.plot(delr / nm, (4 * delr**2) / (lamda), "k")
plt.plot(delr / nm, (4 * delr**2) / (lamda) * np.flip(dat), "k-.")
# plt.plot(NZdw,cut_offs)
plt.xlabel("smallest zone width (nm)")
plt.ylabel("DOF (m)")
plt.tight_layout()
plt.legend(["monochromatic", "polychromatic"])
plt.grid()
plt.xlim(30, 80)
plt.ylim(0, 1e-5)
plt.savefig("szw.png")
# %%

# relationship for DOF to the number of zones
DOF_v = (Dia**2) / (8 * NZ**2)
DOF_mono_fit = CubicSpline(np.flip(DOF_v), np.flip(NZ))
DOF_curve_fit = CubicSpline(NZ, DOF_v * np.flip(fit(NZ / monochr)))
plt.figure()
plt.plot(NZ, DOF_v, "k-.")
plt.plot(NZ, DOF_v * np.flip(fit(NZ / monochr)), "k")
plt.plot(opt_val * monochr, (DOF_curve_fit(opt_val * monochr, nu=0)), "x")

# plt.plot(NZdw,cut_offs)
plt.xlabel("NZ")
plt.ylabel("DOF (m)")
plt.tight_layout()
plt.legend(["monochromatic", "polychromatic"])
plt.grid()
# plt.xlim(30,80)
# plt.ylim(0,1e-5)
plt.savefig("szw.png")

plt.figure()
plt.plot(DOF_v, DOF_mono_fit(DOF_v, nu=0))

zone_eq = DOF_mono_fit(DOF_curve_fit(opt_val * monochr, nu=0))
print(
    f"So while the optimal resolution value returns {round(eq_NZ/monochr,4)}, resulting from optimal value, {round(opt_val,4)},",
    f"The DOF for that optimal zone number is equivalent to {round(zone_eq/monochr,4)}",
)

# %%

# Derivative comparison of the DOF and resolution curves
dev = 1
res_dat = fit_res(cc, nu=dev)
DOF_dat = DOF_fit(cc, nu=dev)
plt.figure()
plt.plot(cc, res_dat / fit_res(1, nu=dev), "k-")
plt.plot(cc, DOF_dat / DOF_fit(1, nu=dev), "k-.")
plt.legend(["res der", "DOF der"])
plt.xlabel("ζ/Nz")
plt.ylabel("normalised derivative")
plt.grid()
plt.tight_layout()


# %%
# Generating the intensity of the focus if the decrease in bandwidth
# corresponds to decreased energy


def Gauss_spec(wav, wav_u, dlamda):
    return np.exp(-((wav - wav_u) ** 2) / (2 * (dlamda) ** 2))


width = np.linspace(nz * 0.3, nz * 3, wl)
FWHMl = lamda / width
dlamda = FWHMl / 2.3548  # (2*np.sqrt(np.log(2)))
length = 10000
wav = np.linspace(lamda - 4 * dlamda[0], lamda + 4 * dlamda[0], length)
dl = 8 * dlamda / length
Fim1 = np.zeros(N // 2) * 0j
plt.figure()
energy = np.zeros(len(width))
poly_old = Gauss_spec(wav, lamda, dlamda[i])
for i in range(len(width)):
    energy[i] = np.sum(Gauss_spec(wav, lamda, dlamda[i])) * dl

lp = len(peaks)
# peak_fit = CubicSpline(width/nz,peaks[lp//2]*energy)
WNZ = np.linspace(width[0] / nz, width[-1] / nz, 1000)
plt.figure()
plt.plot(width / nz, peaks[lp // 2] * energy, "kd")
plt.plot(WNZ, peaks[lp // 2] * energy, "k-")
plt.title("peak intensity for focus")
plt.xlabel("Nz/ζ")
plt.ylabel("intensity [norm.]")
plt.grid()
plt.tight_layout()


# %%
# create a function that recreates the plots based on the NZ/monochr
from scipy.interpolate import RectBivariateSpline

func = RectBivariateSpline(width / nz, ri, plots)
plt.plot(np.transpose(func(width / nz, ri / (1.5 / (lamda * fz / Dia)))))
R, WNZ = np.meshgrid(ri, width / nz)
np.savetxt("width_plots.txt", plots)
plt.figure()
plt.contourf(R, WNZ, np.flipud(plots))
np.savetxt("R.txt", R)
np.savetxt("WNZ.txt", WNZ)
