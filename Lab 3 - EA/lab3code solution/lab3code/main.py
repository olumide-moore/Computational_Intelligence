import pandas as pd
from lib import tsp
from lib import cs3ci

ulysses16_coords = pd.read_csv('ulysses16.csv',header=0).values
tsp_instance = tsp.construct_from_city_coords(ulysses16_coords)
TIME_LIMIT = 3

print('Running an EA for {} seconds'.format(
  TIME_LIMIT))

ea_route, ea_route_cost = cs3ci.ea.solve_tsp(
  tsp_instance,
  pop_size = 10,
  plot_fitness=True,
  time_limit = TIME_LIMIT
)

tsp.display_route(
  tsp_instance,ea_route, city_coords=ulysses16_coords,
  title="EA-generated route on the Ulysses16 TSP Instance.\n" +
    "Cost: {}".format(ea_route_cost)
)