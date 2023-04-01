import time
import random

class Genetic:
    def __init__(self, Target):
        self.Gene_Pool = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\#$%&'()*+,-./:;<=>?@[" ]^_`{|}~'''
        self.Target = Target
        self.pop_size = 1000
        self.Target_len = len(Target)
        self.pop = []
        self.nextgen = []
        self.flag = False
        self.generation = 0

    class Member:
        def __init__(self,chromosome):
            self.chromosome = chromosome
            self.fitness = 0

    def random_gene(self):
        Gene = random.choice(self.Gene_Pool)
        return Gene
    
    def create_chromosome(self):
        chromosome = [self.random_gene() for _ in range(self.Target_len)]
        return chromosome
    
    def calculate_fitness(self):
        for member in self.pop:
            member.fitness = 0
            for i in range(self.Target_len):
                if member.chromosome[i] == self.Target[i]:
                    member.fitness += 1

                if member.fitness == self.Target_len:
                    self.flag = True

    def cross_over(self):
        last_best = int((75 * self.pop_size) / 100)
        self.nextgen = []
        self.nextgen.extend(self.pop[last_best:])
        while True:
            if len(self.nextgen) < self.pop_size:
                parent1 = random.choice(self.pop[last_best:]).chromosome
                parent2 = random.choice(self.pop[last_best:]).chromosome
                child = []
                for gene1,gene2 in zip(parent1,parent2):
                    prob = random.random()
                    if prob < 0.45:
                        child.append(gene1)
                    elif prob < 0.9:
                        child.append(gene2)
                    else:
                        child.append(self.random_gene())   

                self.nextgen.append(self.Member(child)) 
            else:
                break

        self.pop = self.nextgen

    def main(self):
        for _ in range(self.pop_size):
            self.pop.append(self.Member(self.create_chromosome()))

        while not self.flag:
            self.calculate_fitness()
            self.pop = sorted(self.pop, key=lambda member: member.fitness)
            self.cross_over()
            self.generation += 1
            print(f'Generation : {self.generation}')
            print(f'Best Fitness : {self.pop[1].fitness}')
        print(f'You found {self.Target}')
        print(f'Total Generation : {self.generation}')


        
target = "Lorem Ipsum *61891/@"
Go = Genetic(target)
Go.main()