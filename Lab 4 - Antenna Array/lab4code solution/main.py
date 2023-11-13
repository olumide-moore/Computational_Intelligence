from lib.antennaarray import AntennaArray
import pyswarms as ps
import numpy as np

N_ANTENNA = 3
STEERING_ANGLE = 45
POP_SIZE = 20
N_ITERATIONS = 10

# Construct an instance of the antenna array problem with 3 antennae and a
# steering angle of 45 degree.
antenna_array_problem = AntennaArray(N_ANTENNA,45)

# For pyswarms, the cost function should take a 2D array containing the
# positions of all particles as input and should return a list of their costs
# as output.
def constrained_cost(pop):
  # We will assume that the last antenna is fixed in position to meet the 
  # aperture size constraint. We optimise the positions of the remaining
  # N_ANTENNA-1 antennae and add this last antenna in during the evaluation.
  #
  # np.column_stack can add an extra column to a 2D numpy array. We use it to
  # add this last antenna to all positions in the population at once.
  extended_pop = np.column_stack((pop,[N_ANTENNA/2]*POP_SIZE))
  # print(extended_pop)
  # 
  # We then evaluate each member of the population (with the last antenna
  # added to each) and return a list of their fitnesses.
  return [
    antenna_array_problem.evaluate(design)
    for design in extended_pop
  ]

# Create bounds. See: https://pyswarms.readthedocs.io/en/latest/examples/tutorials/basic_optimization.html#Optimizing-a-function-with-bounds
min_bound = 0 * np.ones(N_ANTENNA-1)
max_bound = (N_ANTENNA/2 - 0.25) * np.ones(N_ANTENNA-1)
# print(min_bound, max_bound)
bounds = (min_bound, max_bound)

# Set the PSO social, cognitive and inertial coefficients
options = {'c1': 1.1193, 'c2': 1.1193, 'w':0.721}

# Instantiate PSO
optimizer = ps.single.GlobalBestPSO(
  n_particles=POP_SIZE,
  dimensions=N_ANTENNA-1,
  options = options,
  bounds = bounds
)

# Perform optimization
best_ssl, pos = optimizer.optimize(constrained_cost, iters=N_ITERATIONS)
print("Best peak SLL after {} iterations of PSO: {}".format(
  N_ITERATIONS, best_ssl))