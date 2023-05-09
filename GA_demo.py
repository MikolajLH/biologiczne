import GA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def fitness(coefs : np.array, X, Y):
    P = np.polynomial.Polynomial(coefs)
    return 1 / sum(np.abs(P(X) - Y))

f, a, b, n = np.sin, -2*np.pi, 2*np.pi, 100
X = np.linspace(a, b, n)
Y = f(X)

population_size = 1000
dim = 6
initial_population = [np.random.normal(1, size=dim) for _ in range(population_size)]

evolution_history, statistics = GA.genetic_algorithm(
    initial_population,
    (fitness, X, Y),
    (GA.rank_selection,),
    (GA.double_point_crossover,),
    [(GA.gaussian_multiplicative_mutation, 0.1),
     (GA.gaussian_additive_mutation, 0.1)],
     0.01, 200, lambda f: 1/f)


_, best_f, worst_f, avg_f, Q1_f, median_f, Q3_f, std_f = zip(*statistics)

plt.plot(best_f, label="best")
plt.plot(worst_f, label="worst")
plt.plot(avg_f, label="avg")
plt.plot(median_f, label="median")
plt.plot(std_f, label="stddev")
plt.plot(Q1_f, label="Q1")
plt.plot(Q3_f, label="Q3")
plt.semilogy()
plt.legend()
plt.show()


def draw_frame(frame):
    W = np.linspace(a, b, 1000)
    t, (w, f) = frame
    P = np.polynomial.Polynomial(w)

    plt.cla()
    plt.scatter(X, Y, color='red')
    plt.title(f"Generation {t}, fitness: {f}")
    plt.plot(W, P(W), color='green')

anim = FuncAnimation(
    plt.figure(),
    draw_frame, list(enumerate(evolution_history)), interval = 100, repeat = False
)
plt.show()
