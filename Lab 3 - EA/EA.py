from random import sample, randint, shuffle
import random
from numpy.random import rand, uniform
import time
from pandas import read_csv
from math import sqrt
# from TSPbase import random_route, create_adjcency_matrix, evaluate_fitness

def random_route(tsp_instance):
    """
    param tsp_instance: a given tsp instance
    return: list of random order of cities to visit e.g. [0,1,2,3] or [2,3,1,0] given count=4
    """
    route=list(range(len(tsp_instance)))
    shuffle(route)
    return route

def create_adjcency_matrix(points):
    """
    param points: 2D-list of points in the form [[x1,y1],[x2,y2],...] each representing the x,y coordinate of a city
    return: 2D adjacency matrix i.e. the distance between each city in form [[0,p0-p1,p0-p2,...],[p1-p0,0,p1-p2,...],...,[pn-p0,pn-p1,...,0]]
    """
    n_cities=len(points)
    adjacency_matrix=[[0 for i in range(n_cities)] for j in range(n_cities)] #Create a matrix of size n_cities*n_cities
    for i in range(len(points)): #Populate the matrix with the distances
        for j in range(len(points)):
            if i!=j:
                x=(float(points[j][0])-float(points[i][0]))**2 #(xⱼ-xᵢ)²
                y=(float(points[j][1])-float(points[i][1]))**2 #(yⱼ-yᵢ)²
                adjacency_matrix[i][j]=sqrt(x+y) #√(xⱼ-xᵢ)²+(yⱼ-yᵢ)²
    return adjacency_matrix

def evaluate_fitness(route, adjacency_matrix):
    """
    param route: list of order of cities to visit
    param adjacency_matrix: 2D adjacency matrix of cities i.e. the distance between each city in form [[0,p0-p1,p0-p2,...],[p1-p0,0,p1-p2,...],...,[pn-p0,pn-p1,...,0]]
    return: float/int fitness of the route
    """
    fitness=0 
    for i in range(len(route)-1):  #Loop till the second last city as the last city will be connected to the first city
        fitness+=adjacency_matrix[route[i]][route[i+1]] #Add the distance fitness between the current city and the next city
    fitness+=adjacency_matrix[route[-1]][route[0]] #Add the distance between the last city and the first city
    # return f"Route {route}  fitness: {fitness}"
    return fitness


population=[]

def init_population(population_size):
    population.clear()
    for i in range(population_size):
        population.append(random_route(adjacency_matrix))

def roulette_wheel_selection(population, n_winners):
    '''
    param population: current population
    param n_winners: number of parents to select from the population
    return: list of parents selected by roulette wheel selection
    '''
    selected_parents=[]
    fitness_scores=[evaluate_fitness(route,adjacency_matrix) for route in population]
    cumulative_fitness = [sum(fitness_scores[:i+1]) for i in range(len(fitness_scores))]
    total_fitness=sum(fitness_scores)
    for _ in range(n_winners):
        spin= uniform(0,total_fitness)
        for i, fitness in enumerate(cumulative_fitness):
            if cumulative_fitness[i]>=spin:
                selected_parents.append(population[i])
                break
    return selected_parents



def rank_based_selection(population, n_winners):
    '''
    The rank based selection selects n_winners by ranking the population by fitness and selecting parents by their rank
    return: list of parents selected by rank based selection
    '''
    n=len(population)
    ranked_population=sorted(population, key=lambda x: evaluate_fitness(x,adjacency_matrix), reverse=True) #Sort the population by fitness with the best fitness first
    ranks=list(range(1,n+1)) #List of ranks
    c=2
    probabilities=[(c - (c*(i / n))) / n for i in ranks] #Calculate the probability of each rank
    return random.choices(ranked_population, weights=probabilities, k=n_winners) #Select n_winners from parents using the ranked population with set probability of each rank
    
    
    # probabilities=[((2 - s)/n) + (2*i*(s - 1)) / (n * (n - 1)) for i in ranks] 




def tournament_selection(population, n_winners):
    '''
    The tournament selection selects N_OFFSPRING number of parents by randomly selecting 10 routes(subset) from the population and selecting the best route from the subset as the parent
    param population: current population
    param n_winners: number of parents to select from the population
    return: list of parents selected by tournament
    '''
    selected_parents=[]
    for _ in range(n_winners):
        subset=sample(population,k=10)
        best_fitness=float('inf')
        for parent in subset:
            fitness=evaluate_fitness(parent,adjacency_matrix)
            if fitness<best_fitness: best_fitness,best_parent=fitness,parent
        selected_parents.append(best_parent)
    return selected_parents

def swap_mutation(route,probability):
    if rand()<probability:
        i,j=sample(range(len(route)),2)
        route[i],route[j]=route[j],route[i]
    return route


def order_one_crossover(parent1,parent2,probability):
    tour_length=len(parent1)
    #Choose two random crossover points
    crossover_point1=randint(0,tour_length-1)
    crossover_point2=randint(crossover_point1+1,tour_length)
    # print(crossover_point1,crossover_point2)
    if rand()<probability:
        offspring1=[None for _ in range(tour_length)]#Initialise offspring with None
        offspring2=[None for _ in range(tour_length)]
        #Copy the swath from parent1 to offspring1
        offspring1[crossover_point1:crossover_point2]=parent1[crossover_point1:crossover_point2]
        #Copy the swath from parent2 to offspring2
        offspring2[crossover_point1:crossover_point2]=parent2[crossover_point1:crossover_point2]
        #Fill the rest of the offspring from the other parent
        cur_index=crossover_point2%tour_length
        while cur_index!=crossover_point1:
            for par,offsp in [(parent2,offspring1),(parent1,offspring2)]:
                i=cur_index
                while par[i] in offsp:
                    i=(i+1)%tour_length
                offsp[cur_index]=par[i]
            cur_index=(cur_index+1)%tour_length
        return offspring1,offspring2
    return parent1,parent2     


# adjacency_matrix=  [[0, 20, 42, 35],
#                     [20, 0, 30, 34],
#                     [42, 30, 0, 12],
#                     [35, 34, 12, 0]]
# ulysses16_coords = read_csv('ulysses16.csv',header=0).values

cities82_9= read_csv('cities82_9.csv',header=0).values
adjacency_matrix= create_adjcency_matrix(cities82_9) #Create adjancency matrix of points


def EA(population_size, N_OFFSPRING, mutation_probability=0.7, crossover_probability=0.2,time_limit=3):
    init_population(population_size)
    best_fitness=float('inf')
    best_route=None
    start=time.time()
    while time.time()-start<time_limit:
        # parents=tournament_selection(population, N_OFFSPRING)
        parents=rank_based_selection(population, N_OFFSPRING)
        children=[]
        for i in range(0,N_OFFSPRING,2):
            o1,o2=order_one_crossover(parents[i],parents[i+1],crossover_probability)
            o1=swap_mutation(o1,mutation_probability)
            o2=swap_mutation(o2,mutation_probability)
            for offspring in [o1,o2]:
                fitness=evaluate_fitness(offspring,adjacency_matrix)
                if fitness<best_fitness: #If the offspring fitness is better than the current best fitness, update the best fitness and best route
                    print(best_fitness)
                    best_fitness=fitness
                    best_route=offspring
            children.append(o1)
            children.append(o2)
        population.clear()
        population.extend(children)
    return best_route, best_fitness

print(EA(50,50))
# mutated_parents=[swap_mutation(p) for p in selected_parents]

##Lowest i got - 73.99982631274878
##Suggested answer with 16 cities is 73.98


##Mutation probability
##With an increase mutation probability, the algorithm is able to find a better solution 
##as opposed to a lower mutation probability. This is because with a higher mutation probability,
##the offsprings are likely to be more different from the parents (new phenotypes are more likely to be generated), 
##which allows the algorithm to explore a larger search space.

##Crossover probability
##With an increase crossover probability, the algorithm is able to find a better solution
##as opposed to a lower crossover probability. This is because with a higher crossover probability,
##there is a higher chance of generating offsprings, which allows the algorithm to explore a larger search space.