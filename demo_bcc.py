from lib import *
import scipy.constants as const
import time

timestamp = str(int(time.time()))

supercell_size = 3
T = 1000.
rate = 4.6e12 * np.exp( -.62*const.e / (const.k*T) )
num_vacs = 1
lattice_constant = 4e-10
Z = 8

num_runs =  100
max_num_steps = 100000
time_cutoff = 1e-9
num_steps = int(time_cutoff * rate * num_vacs * Z * 1.5)# factor 1.5 for tolerance
num_steps = num_steps if num_steps < max_num_steps else max_num_steps
jumps = [
    [1, 1,1],[1, 1,-1],
    [1,-1,1],[1,-1,-1],
    [-1,1, 1],[-1, 1,-1],
    [-1,-1,1],[-1,-1,-1]
]
rates = [
    rate, rate, rate, rate,
    rate, rate, rate, rate,
    rate, rate, rate, rate
]
mylat = Lattice([1, 0, 0, 0, 0, 0, 1, 0], jumps, rates, supercell_size, supercell_size, supercell_size, num_vacs, lattice_constant)
print (mylat.lattice)
mykmc = kMC(mylat)

t_list = []
msd_list = []
for i in range(num_runs):
    print (str(i) + ". run")
    mykmc.simulate(num_steps)
    t_list.append(mykmc.t)
    msd_list.append(mykmc.particle_msd)

out_list = []
for t, msd in zip(t_list, msd_list):
    out_list += [t, msd]
supercell_string = str(supercell_size) + 'x' + str(supercell_size) + 'x' + str(supercell_size)
np.save('bcc_msd_data_' + str(num_runs) + 'runs_' + supercell_string + '_cell_tsim' + '{:1.0e}'.format(time_cutoff) + '_'  + str(num_vacs) + 'vO_' + str(int(T)) + 'K_' + timestamp + '.npy', np.array(out_list))