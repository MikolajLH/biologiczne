import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation



class ParameterizedFunction:
    def __init__(self) -> None:
        self.fs = []
        self.ps = []
        self.fitness = 0

    def __call__(self, x):
        if len(self.ps) == 0 or len(self.fs) == 0:
            raise RuntimeError
        
        res = None 

        for p, f in zip(self.ps,self.fs):
            if res is None:
                res = p * f(x)
            else:
                res += p * f(x)
        return res
    
    def add(self, f):
        self.fs.append(f)
        self.ps.append(1.)

    def randomize_u(self, a, b):
        for i in range(len(self.ps)):
            self.ps[i] = np.random.uniform(a,b)
    
    def mutate_u(self, a, b):
        i = np.random.randint(0, len(self.ps))
        self.ps[i] += np.random.uniform(a,b)



def merge(m : ParameterizedFunction,f : ParameterizedFunction):
    if len(m.ps) != len(f.ps):
        raise RuntimeError()
    
    child = ParameterizedFunction()
    child.fs = list(m.fs)
    child.ps = list(m.ps)

    for i in range(len(m.ps)):
        if np.random.binomial(1,0.5) == 1:
            child.ps[i] = m.ps[i]
        else:
            child.ps[i] = f.ps[i]

    return child




if __name__ == "__main__":
    f, a, b, n = np.cos, 0, 2 *np.pi, 10
    elite_percent = 0.10
    mutation_chance = 1.
    pop_size = 1000
    number_of_generations = 100

    X = np.linspace(a,b,n)
    Y = f(X)

    bests = []

    population = [ ParameterizedFunction() for _ in range(pop_size)]
    for pf in population:
        pf.add(np.vectorize(lambda x: 1.))
        pf.add(np.vectorize(lambda x: x))
        pf.add(np.vectorize(lambda x: x*x))
        pf.add(np.vectorize(lambda x: x*x*x))
        pf.add(np.vectorize(lambda x: x*x*x*x))

        pf.randomize_u(-100, 100)

    for t in range(number_of_generations):
        
        #calculating fitness
        for pf in population:
            nY = pf(X)
            pf.fitness = np.sum(np.abs(nY - Y))

        population.sort(key = lambda e: e.fitness)
        print(f"generation: {t}, pop_size: {len(population)} best_fitness {population[0].fitness}")
        
        bests.append(copy.deepcopy(population[0]))

        #creating mating pool
        elite_size = int(pop_size * elite_percent)
        population = population[:elite_size]

        #crossover and mutation
        children = []
        for _ in range(pop_size - elite_size):
            M, F = np.random.choice(population, 2, replace=False)
            children += [merge(M,F)]
            if np.random.binomial(1,mutation_chance):
                pf.mutate_u(-10, 10)

        population += children
    
    for pf in population:
            nY = pf(X)
            pf.fitness = np.sum(np.abs(nY - Y))

    population.sort(key = lambda e: e.fitness)
    best = population[0]
    print(f"best_fitness {best.fitness}")


    
    W = np.linspace(a, b, 1000)

    def draw_frame(frame):
        t, pf = frame
        plt.cla()
        plt.scatter(X, Y, color='red')
        plt.title(f"Generation {t}, fitness: {pf.fitness}")
        plt.plot(W, pf(W), color='green')
        #plt.scatter(X, pf(X))

    temp = list(enumerate(bests))

    anim = animation.FuncAnimation(
        plt.figure(),
        draw_frame, temp, interval = 100, repeat = False
    )

    plt.show()






