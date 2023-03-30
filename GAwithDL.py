import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten input images
x_train = x_train.reshape((-1, 28 * 28))
x_test = x_test.reshape((-1, 28 * 28))

# Define the fitness function
def fitness(params):

    # Create a neural network model
    model = Sequential()
    model.add(Dense(int(params[0]), activation='relu', input_shape=(784,)))
    for i in range(int(params[1])):
        model.add(Dense(int(params[2]), activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params[3]),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(x_train, y_train, batch_size=int(params[4]), epochs=int(params[5]), verbose=0)

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy

# Define the search space
search_space = [(4, 8), # number of neurons in the first hidden layer
                (0, 3), # number of hidden layers
                (4, 8), # number of neurons in each hidden layer
                (0.001, 0.1), # learning rate
                (64, 128), # batch size
                (5, 10)] # number of epochs

# Define the population size and number of generations
pop_size = 20
num_generations = 10

def genetic():

    # Initialize the population
    population = []
    for i in range(pop_size):
        params = []
        for space in search_space:
            params.append(random.uniform(space[0], space[1]))
        population.append(params)

    # Initialize lists to store best fitness scores
    best_fitness_scores_history = []
    times = []
    start_time = time.time()

    # Iterate over generations
    for gen in range(num_generations):

        print(f'Generation {gen + 1}')

        # Evaluate fitness of each individual in the population
        fitness_scores = []
        for individual in population:
            fitness_scores.append(fitness(individual))

        # Normalize fitness scores
        sum_fitness = np.sum(fitness_scores)
        fitness_probs = [score/sum_fitness for score in fitness_scores]

        # Select parents for reproduction
        parents = []
        for i in range((pop_size // 2)):
            idx1 = np.random.choice(range(pop_size), size=1, p=fitness_probs)[0]
            idx2 = np.random.choice(range(pop_size), size=1, p=fitness_probs)[0]
            parents.append((population[idx1], population[idx2]))

        # Reproduce new offspring
        offspring = []

        for parent1, parent2 in parents:

            child1, child2 = [], []

            for i in range(len(parent1)):
                if random.random() < 0.5:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
                else:
                    child1.append(parent2[i])
                    child2.append(parent1[i])
                
            offspring.append(child1)
            offspring.append(child2)
        
        # Mutate some of the offspring
        for i in range(len(offspring)):
            for j in range(len(offspring[i])):
                if random.random() < 0.01:
                    space = search_space[j]
                    offspring[i][j] = random.uniform(space[0], space[1])

        # Replace the old population with the new offspring
        population = offspring

        # Evaluate fitness of final population
        fitness_scores = []
        for individual in population:
            fitness_scores.append(fitness(individual))

        # Select the best individual as the result
        best_idx = np.argmax(fitness_scores)
        best_individual = population[best_idx]
        best_fitness_score = fitness_scores[best_idx]
        print('Best individual:', best_individual)
        print(f'Best fitness score in generation {gen + 1}: {best_fitness_score}')

        # Append best fitness score to history lists
        best_fitness_scores_history.append(best_fitness_score)
        times.append(time.time() - start_time)

        print(f'Time : {time.time() - start_time}')
        start_time = time.time()

    # Plot best fitness scores over time
    plt.figure(figsize=(8, 6))
    plt.plot(best_fitness_scores_history, label='Best Fitness Scores')
    plt.scatter(np.argmax(best_fitness_scores_history), np.max(best_fitness_scores_history), color='red', label='Max Fitness Score')
    plt.scatter(np.argmin(best_fitness_scores_history), np.min(best_fitness_scores_history), color='green', label='Min Fitness Score')
    plt.plot(np.ones(num_generations) * np.mean(best_fitness_scores_history), label='Mean Fitness Score')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title('Best Fitness Scores over Generation')
    plt.legend()
    plt.savefig('./Best_Fitness_Scores_over_Generation.png')
    plt.show()

    # Plot generations and time
    plt.figure(figsize=(8, 6))
    plt.plot(times, label='Time (s)')
    plt.scatter(np.argmax(times), np.max(times), color='red', label='Max Time')
    plt.scatter(np.argmin(times), np.min(times), color='green', label='Min Time')
    plt.plot(np.ones(num_generations) * np.mean(times), label='Mean Time')
    plt.xlabel('Generation')
    plt.ylabel('Time (s)')
    plt.title('Generations and Time')
    plt.legend()
    plt.savefig('./Generations_and_Time.png')
    plt.show()

genetic()