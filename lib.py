import numpy as np
import matplotlib.pyplot as plt

''' Lattice defined with asymmetric unit '''
class Lattice:
    def __init__(self, sites, jumps, rate_constants, Nx, Ny, Nz, N_vac, lattice_constant=1., keep_full_history=False):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.N_vac = N_vac
        self.sites = sites
        self.jumps = np.array(jumps)
        self.rate_constants = np.array(rate_constants)
        self.lattice = np.zeros((2*self.Nx, 2*self.Ny, 2*self.Nz), dtype=np.int32)
        self.lattice_constant = lattice_constant
        self.keep_full_history = keep_full_history
        self.create_occupation()
    def create_occupation(self):
        ''' fill the virtual sites with the given configuration of the asymmetric unit.
            In the current implementation, only one particle species can be implemented,
            any number >0 will be interpreted as a particle of that species. '''
        for nx in range(self.Nx):
            for ny in range(self.Ny):
                for nz in range(self.Nz):
                    self.lattice[2*nx,   2*ny,   2*nz] =   self.sites[0]
                    self.lattice[2*nx+1, 2*ny,   2*nz] =   self.sites[1]
                    self.lattice[2*nx+1, 2*ny+1, 2*nz] =   self.sites[2]
                    self.lattice[2*nx,   2*ny+1, 2*nz] =   self.sites[3]
                    self.lattice[2*nx,   2*ny, 2*nz+1] =   self.sites[4]
                    self.lattice[2*nx+1, 2*ny, 2*nz+1] =   self.sites[5]
                    self.lattice[2*nx+1, 2*ny+1, 2*nz+1] = self.sites[6]
                    self.lattice[2*nx,   2*ny+1, 2*nz+1] = self.sites[7]
        ''' fill part of the actual sites with vacancies.
            Vacancies will be denoted by negative indices (-1, -2, -3, ... -N_vac). '''
        site_ids = np.where(self.lattice != 0)
        self.N_sites = site_ids[0].size
        self.N_particles = self.N_sites - self.N_vac
        if self.N_vac < self.N_sites:
            count_vacs = 0
            while count_vacs<self.N_vac:
                x, y, z = np.random.randint(2*self.Nx), np.random.randint(2*self.Ny), np.random.randint(2*self.Nz)
                if self.lattice[x, y, z] > 0:
                    self.lattice[x, y, z] = -(count_vacs+1)
                    count_vacs += 1
        else:
            print ("Error: no. of vacancies must be smaller than number of sites!")
        ''' assign a unique label (1, 2, 3, ... N_particles) to the particles. '''
        particle_ids = np.where(self.lattice > 0)
        self.lattice[particle_ids] = np.arange(1, self.N_particles+1)
        ''' create displacement array '''
        self.reset_displacements()
    def reset_displacements(self):
        self.displacements = np.zeros((self.N_particles, 3))
        self.vacancy_displacements = np.zeros((self.N_vac, 3))
        self.particles_num_jumps = np.zeros(self.N_particles)
        self.vacancies_num_jumps = np.zeros(self.N_vac)
        self.displacement_history = []
        self.full_displacement_history = []
    def periodic_boundaries(self, vec):# this implementation works only if there are no [2,0,0] (or longer) jumps!
        x, y, z = vec[0], vec[1], vec[2];
        x_ = 0 if x>=2*self.Nx else 2*self.Nx-1 if x<0 else x
        y_ = 0 if y>=2*self.Ny else 2*self.Ny-1 if y<0 else y
        z_ = 0 if z>=2*self.Nz else 2*self.Nz-1 if z<0 else z
        return np.array([x_, y_, z_])
    def get_all_jumps(self):
        vacancy_ids = np.where(self.lattice < 0)
        vacancy_id_list = []
        direction_list = []
        rate_list = []
        for x, y, z in zip(vacancy_ids[0], vacancy_ids[1], vacancy_ids[2]):
            loc = np.array([x, y, z])
            for jump, rate in zip(self.jumps, self.rate_constants):
                loc_ = self.periodic_boundaries(loc + jump)
                if self.lattice[loc_[0], loc_[1], loc_[2]]>0:# jump is possible!
                    direction_list.append(jump)
                    rate_list.append(rate)
                    vacancy_id_list.append(loc)
        self.jump_locs = np.array(vacancy_id_list)
        self.jump_directions = np.array(direction_list)
        return np.array(rate_list)
    def perform_step(self, id):
        loc = self.jump_locs[id]
        loc_ = self.periodic_boundaries(loc + self.jump_directions[id])
        ''' update particle & vacancy displacements. '''
        particle_label = self.lattice[loc_[0], loc_[1], loc_[2]]
        vacancy_label = self.lattice[loc[0], loc[1], loc[2]]
        self.displacements[particle_label-1] -= self.jump_directions[id]
        self.vacancy_displacements[-vacancy_label-1] += self.jump_directions[id]
        self.particles_num_jumps[particle_label-1] += 1
        self.vacancies_num_jumps[-vacancy_label-1] += 1
        if self.keep_full_history:
            self.full_displacement_history.append(self.lattice_constant/2. * np.copy(self.displacements))
        ''' update lattice configuration. '''
        tmp = self.lattice[loc[0], loc[1], loc[2]]
        self.lattice[loc[0], loc[1], loc[2]] = self.lattice[loc_[0], loc_[1], loc_[2]]
        self.lattice[loc_[0], loc_[1], loc_[2]] = tmp
    def calc_msd(self, calc_num_jumped_particles=False):
        # MSD must be calculated with half lattice constant because of unit cell subdivision (2x2x2)!
        particle_msd = (self.lattice_constant/2.)**2*(self.displacements**2).sum() / self.N_particles
        vacancy_msd =  (self.lattice_constant/2.)**2*(self.vacancy_displacements**2).sum() / self.N_vac
        if calc_num_jumped_particles:
            return particle_msd, vacancy_msd, (self.particles_num_jumps>0).sum(), (self.vacancies_num_jumps>0).sum()
        else:
            return particle_msd, vacancy_msd
    def update_rates(self, rate_constants):
        self.rate_constants = np.array(rate_constants)
    def dump_full_displacement_history(self, filename):
        np.save(filename, self.full_displacement_history)
class kMC:
    def __init__(self, lattice):
        self.lattice = lattice
    def kMC_step(self, i=0, calc_num_jumped_particles=False):
        if ((i+1)%1000==0):
            print (i)
        self.rates = self.lattice.get_all_jumps()
        self.cumulated_rates = self.rates.cumsum()# cumulated sum of rates
        u = np.random.rand()
        uQ = u*self.cumulated_rates[-1]
        id = np.searchsorted(self.cumulated_rates, uQ)
        self.lattice.perform_step(id)
        ''' calculate time step '''
        ut = np.random.rand()
        Delta_t = -np.log(ut) / self.cumulated_rates[-1]
        if calc_num_jumped_particles:
            particle_msd, vacancy_msd, particles_num_jumps, vacancies_num_jumps = self.lattice.calc_msd(calc_num_jumped_particles)
            return [Delta_t, particle_msd, vacancy_msd, particles_num_jumps, vacancies_num_jumps]
        else:
            particle_msd, vacancy_msd = self.lattice.calc_msd(calc_num_jumped_particles)
            return [Delta_t, particle_msd, vacancy_msd]
    def simulate(self, N_steps, calc_num_jumped_particles=False, reset_displacements=True):
        if reset_displacements:
            self.lattice.reset_displacements()
        results = np.array([self.kMC_step(i, calc_num_jumped_particles) for i in range(N_steps)]).T
        self.Delta_t = results[0]
        self.particle_msd = results[1]
        self.vacancy_msd = results[2]
        if calc_num_jumped_particles:
            self.particles_jumped = results[3]
            self.vacancies_jumped = results[4]
        self.t = np.cumsum(self.Delta_t)