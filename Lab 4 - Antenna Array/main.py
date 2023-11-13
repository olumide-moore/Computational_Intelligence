from antennaarray import AntennaArray
import random
import time
# random.seed(42)
count=0
class Particle:
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance
        self.bounds=self.problem_instance.bounds()
        self.position = self.constrained_random_parameters()
        self.velocity = self.generate_first_velocity()
        self.max_velocity = [0.2*(high-low) for [low, high] in self.bounds]
        self.personal_best_position = self.position
        self.personal_best_cost = self.problem_instance.evaluate(self.position)

    def constrained_random_parameters(self): 
        while True:
            design=[low + random.random()*(high-low) for [high, low] in self.bounds]
            design[-1] = self.problem_instance.n_antennae/2
            if self.problem_instance.is_valid(design):
                return design
    
    def generate_first_velocity(self):
        #Generate a random feasible position
        feasible_position = self.constrained_random_parameters()
        #Generate a velocity that is half the difference between first position and a second feasible position
        return [abs((p1-p2)/2) for p1, p2 in zip(self.position, feasible_position)]

    def update_position(self):
        new_position = [p + v for p, v in zip(self.position, self.velocity)]
        if self.problem_instance.is_valid(new_position):
            self.position = new_position
            # print(self.position)
        # else:
            # global count
            # count =count+1
            # # print(count)
            # print(new_position,self.velocity)
            # self.position = self.constrained_random_parameters()
            # self.velocity = self.generate_first_velocity()
        cost = self.problem_instance.evaluate(self.position)
        if (cost) < (self.personal_best_cost):
            # print(cost)
            self.personal_best_cost = cost
            self.personal_best_position = self.position

    def update_velocity(self, global_best_position, inertia, phi_p, phi_g):
        '''phi_p : float cognitive parameter *phi_g : float social parameter *inertia: float inertia parameter bounds'''
        new_velocity = []
        for p, v, p_best, g_best in zip(self.position, self.velocity, self.personal_best_position, global_best_position):
            r_p = random.uniform(0, 1)
            r_g = random.uniform(0, 1)
            new_v = inertia*v + phi_p*r_p*(p_best - p) + phi_g*r_g*(g_best - p)
            new_velocity.append(new_v)
        self.velocity = new_velocity
    
    #update velocity and clamp it to max velocity
    def clamp_velocity(self, global_best_position, inertia, phi_p, phi_g):
        '''phi_p : float cognitive parameter,  phi_g : float social parameter,  inertia: float inertia parameter bounds'''
        new_velocity = []
        for p, v, p_best, g_best, max_v in zip(self.position, self.velocity, self.personal_best_position, global_best_position, self.max_velocity):
            r_p = random.uniform(0, 1)
            r_g = random.uniform(0, 1)
            new_v = inertia*v + phi_p*r_p*(p_best - p) + phi_g*r_g*(g_best - p)
            new_v = max(min(new_v, max_v), -max_v) #clamp velocity
            new_velocity.append(new_v)
        self.velocity = new_velocity
    

class Swarm:
    def __init__(self, problem_instance, n_particles, inertia, phi_p, phi_g):
        self.problem_instance = problem_instance
        self.n_particles = n_particles
        self.inertia = inertia
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.particles = [Particle(problem_instance) for _ in range(n_particles)]
        # print(list(map(lambda x: x.position, self.particles)))
        # print(list(map(lambda x: sum(x.position), self.particles)))
        self.global_best_position = self.get_global_best_position()
        self.global_best_cost = self.problem_instance.evaluate(self.global_best_position)

    def get_global_best_position(self):
        return min(self.particles, key=lambda particle: particle.personal_best_cost).personal_best_position
    def update_global_best(self):
        new_global_best_position = self.get_global_best_position()
        new_global_best_cost = self.problem_instance.evaluate(new_global_best_position)
        if new_global_best_cost < self.global_best_cost:
            print(new_global_best_cost)
            self.global_best_cost = new_global_best_cost
            self.global_best_position = new_global_best_position
    def update_particles(self):
        for particle in self.particles:
            particle.update_velocity(self.global_best_position, self.inertia, self.phi_p, self.phi_g)
            # particle.clamp_velocity(self.global_best_position, self.inertia, self.phi_p, self.phi_g)
            particle.update_position()
        #Update global best
        self.update_global_best()

def pso(problem_instance, n_particles, time_limit, inertia, phi_p, phi_g):
    start=time.time()
    swarm = Swarm(problem_instance, n_particles, inertia, phi_p, phi_g)
    while time.time()-start < time_limit:
        swarm.update_particles()
    return swarm.global_best_position, swarm.global_best_cost      

# Construct an instance of the antenna array problem with 3 antennae and a
# steering angle of 45 degree.
N_ANTENNA = 4
STEERING_ANGLE = 35
POP_SIZE = 10
TIME_LIMIT = 30
antenna_array_problem = AntennaArray(N_ANTENNA,STEERING_ANGLE)

best_ssl=pso(antenna_array_problem, POP_SIZE, TIME_LIMIT, 0.721, 1.1193, 1.1193)
print("Best peak SLL after {} 8s based on PSO initialisation: {}".format(
  TIME_LIMIT, best_ssl))




# print(count)
###The velocity clamping slows down the exploration of the search space which gives it opportunity to exploit the search space more. 
# This is because the velocity is clamped to a maximum velocity which is a fraction of the search space. Although, it causes the particles to get stuck in local optima due to the lack of exploration.

 
# print(antenna_array_problem.evaluate([0.5, 1.0, 1.5]))

# Best peak SLL after 8s based on PSO initialisation:     ##3 antennaes 20particles
# ([0.8458673921403798, 0.2597087374472885, 1.5], -12.064639976528488)

#Best according to others is -12.01

# A simple example of how the AntennaArray class could be used as part
# of a random search.


# def random_parameters(antenna_array_problem):
#     b = antenna_array_problem.bounds()  
#     return [low + random.random()*(high-low) for [high, low] in b]

# ###############################################################################
# # NOTE: This attempt at solving the problem will work really badly! We        #
# # haven't taken constraints into account when generating random parameters.   #
# # The probability of randomly generating a design which meets the aperture    #
# # size constraint is close to zero. This is just intended as an illustration. #
# ###############################################################################

# # # Generate N_TRIES random parameters and measure their peak SLL on the problem,
# # # saving the best parameters.
# best_parameters = random_parameters(antenna_array_problem)
# best_sll = antenna_array_problem.evaluate(best_parameters)
# for _ in range(N_ITERATIONS - 1):
#     parameters = random_parameters(antenna_array_problem)
#     # Note: in this example we are not testing parameters for validity. The
#     # evaluate function penalises invalid solutions by assigning them the
#     # maximum possible floating point value.
#     sll = antenna_array_problem.evaluate(parameters)
#     if sll < best_sll:
#         best_sll = sll
#         best_parameters = parameters

# # print("Best peak SLL after {} iterations based on random initialisation: {}".format(
# #   N_TRIES, best_sll))
  
# ###############################################################################
# # How can we improve on the above attempt? By trying to generate initial      #
# # parameters which meet the aperture size constraint!                         #
# ###############################################################################

# def constrained_random_parameters(antenna_array_problem):
#     b = antenna_array_problem.bounds()  
#     design = [low + random.random()*(high-low) for [high, low] in b]
#     design[-1] = antenna_array_problem.n_antennae/2
#     return design
    
# # Try random search again with this new method of generating parameters.
# best_parameters = constrained_random_parameters(antenna_array_problem)
# best_sll = antenna_array_problem.evaluate(best_parameters)
# for _ in range(N_ITERATIONS - 1):
#     parameters = constrained_random_parameters(antenna_array_problem)
#     sll = antenna_array_problem.evaluate(parameters)
#     if sll < best_sll:
#         best_sll = sll
#         best_parameters = parameters
