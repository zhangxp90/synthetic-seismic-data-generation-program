# A program to generate seismic data containing hyperbolas
# writen by ZXP on 20181108

import numpy as np
import math
import os
import csv
import random
from array import *


"""parameters"""
Generate_Num = 8    #Number of samples

dt = 0.01   #sampling interval in secs
f0 = [20, 50]   #central freq. of a Ricker wavelet in Hz
tmax = 1.28     #maximun time of the simulation in secs
i_h = range(-315, 321, 10)  #vector of offsets in meters
tau = [0.02, 0.7, 1.1]        #vectors of intercept (in secs)
v = [500,1800,3800]           #vectors of the velocities (in m/s)
amp = [5.0, 4.8,-4.3]           #vectors of the amplitudes of each linear event
rand_range = 0.2

"""basic function"""
def ricker(r_f0, r_dt):
    nw = 2.2/r_f0/r_dt
    nw = 2*math.floor(nw/2)+1
    r_w = np.zeros((nw,1))
    nc = np.ones((nw,1))*math.floor(nw/2)
    k = np.array(range(1, nw+1)).reshape((nw,1))
    alpha = (nc-k+np.ones((nw,1)))*(r_f0*r_dt*math.pi)
    beta=alpha**2
    r_w = np.multiply((np.ones((nw,1)) - beta*2), math.e**(np.zeros((nw,1)) - beta))
    return r_w

def generate_simu_data(_tau, _i_h, _f0, _dt, _tmax, _v, _amp):
    """configuration"""
    n_events = len(_tau)
    nh = len(_i_h)
    h = np.array(_i_h).reshape((1,nh))
    wavelet = ricker(_f0, _dt)
    nw = len(wavelet);
    #plt.plot(range(nw), wavelet)
    #plt.show()
    _nt = math.ceil(_tmax / _dt)
    nfft = 4*(2**math.ceil(math.log(_nt,2)))
    W = np.fft.fft(wavelet,nfft,0)
    D = np.zeros((nfft,nh),dtype=complex)
    i = np.complex(0,1)
    """constitute the image in a raw-by-raw way"""
    # Important: the following lines is to have the maximum of the Ricker
    # wavelet at the right intercept time
    delay = _dt*(math.ceil(nw/2))
    for ifreq in range(0, int(nfft/2)+1):
        w = 2*math.pi*ifreq/nfft/_dt
        for k in range(0, n_events):
            Shift = math.e**(-i*w*((_tau[k]**2*np.ones((1,nh)) + (h/_v[k])**2)**0.5 - delay*np.ones((1,nh))))
            D[ifreq,:] = D[ifreq,:] +_amp[k]* W[ifreq]*Shift
    # Apply w-domain symmetries
    for ifreq in range(1,int(nfft/2)+1):
        D[nfft-ifreq,:] = D[ifreq,:].conjugate()
    d = np.fft.ifft(D,nfft,0)
    f_d = d[0:_nt,:].real
    return _nt, f_d

def generate_rand_parameter(_list):
    ret = []
    for ele in _list:
        ret.append(ele+ele*rand_range*random.uniform(-1,1))
    return ret

def main():
    save_path = os.getcwd()+"/data/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    gen_inductor = range(1024)
    for j in range(Generate_Num):
        #generating the parameters for each sample
        f0_e = f0[0] + random.uniform((f0[1] - f0[0]) / Generate_Num * j, (f0[1] - f0[0]) / Generate_Num * (j+1))
        tau_e = generate_rand_parameter(tau)
        v_e = generate_rand_parameter(v)
        amp_e = generate_rand_parameter(amp)
        #generating data
        nt, simu_data= generate_simu_data(tau_e, i_h, f0_e, dt, tmax, v_e, amp_e)
        #saving data as a csv data
        with open(save_path+'simulating_data_'+str(j)+'.csv', 'w', newline='') as save_file:
            wr = csv.writer(save_file, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
            for i in range(nt):
                wr.writerow(list(simu_data[i,:]))

if __name__ == '__main__':
    main()
