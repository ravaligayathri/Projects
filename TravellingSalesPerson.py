
"""
Created on Thu Oct 11 11:44:46 2018

@author: Ravali Kuppachi
"""
import random

from deap import base
from deap import creator
from deap import tools
import pandas as pd
import statistics

creator.create("TSPMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.TSPMin)

toolbox = base.Toolbox()

# Attribute generator
#
toolbox.register("attr_bool", random.sample, range(8), 8)

# Structure initializers
#
toolbox.register("individual", tools.initIterate, creator.Individual,
    toolbox.attr_bool)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Assign numbers to column names in a dictionary

dic = {}

count=-1
dist=pd.read_csv("TS_Distances_Between_Cities.csv")

for col in dist.columns.values:
    dic[count]=col
    count+=1


distances = pd.read_csv("TS_Distances_Between_Cities.csv",header=None, skipfooter=1,skiprows=1,usecols=range(1,9))



# the goal ('fitness') function to be maximized
def evalTSP(individual):
    dis=[]

    for i in range(0,len(individual)-1):
        new_dis=distances.iloc[individual[i],individual[i+1]]
        dis.append(new_dis)
    return sum(dis),

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", evalTSP)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

# register the crossover operator
toolbox.register("mate", tools.cxOrdered)

# register a mutation operator with a probability to
# shuffle randomly with probability of 0.05
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)



#----------

def main():
    random.seed(64)

    # create an initial population of 500 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=500)
    #print(pop)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.3, 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    #print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    info_path = 'Ravali_Kuppachi_GA_TS_Info.txt'
    out = open(info_path,'w')

    # Begin the evolution
    while min(fits) > 10000 and g < 100:
        # A new generation
        g = g + 1


        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        #print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]


        out.write(str(g)+'. '+ 'Population Size : '+str(len(pop))+' for iteration '+str(g)+'\n')
        out.write('Average Fitness Score : '+str(statistics.mean(fits))+'\n' )
        out.write('Median Fitness Score : '+str(statistics.median(fits))+'\n' )
        out.write('STD of Fitness Score : '+str(statistics.stdev(fits))+'\n' )
        out.write('Size of selected subset of population : '+str(len(invalid_ind))+'\n' )



    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    output_path = 'Ravali_Kuppachi_GA_TS_Result.txt'
    out1 = open(output_path,'w')
    counter=1
    for ind in best_ind:

        out1.write(str(counter)+'. '+str(ind) + '/' + str(dic[ind])+'\n')
        counter+=1

if __name__ == "__main__":
    main()
