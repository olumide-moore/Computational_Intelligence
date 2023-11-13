
from math import sqrt
from random import shuffle

def random_route(count):
    """
    param count: int number of cities
    return: list of random order of cities to visit e.g. [0,1,2,3] or [2,3,1,0] given count=4
    """
    route=[_ for _ in range(count)]
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