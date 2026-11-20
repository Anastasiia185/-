import random
import matplotlib.pyplot as plt

X_MIN, X_MAX = -1.0, 5.0
Y_MIN, Y_MAX =  0.0, 4.0
LX, LY = 6, 6
POP_SIZE    = 50
GENERATIONS = 500
P_CROSS     = 0.9
P_MUT       = 0.01

random.seed(42)

# Декодування 
def decode(bits, a, b, L):
    v = int(bits, 2)
    return a + v * (b - a) / ((1 << L) - 1)


# Фітнес-функція 
def evaluate(ind):
    xb, yb = ind
    x = decode(xb, X_MIN, X_MAX, LX)
    y = decode(yb, Y_MIN, Y_MAX, LY)
    return 2 * x + y, x, y


# Схрещування
def crossover(p1, p2):
    xb1, yb1 = p1
    xb2, yb2 = p2
    if random.random() < P_CROSS:
        pt = random.randint(1, LX - 1)
        xb1, xb2 = xb1[:pt] + xb2[pt:], xb2[:pt] + xb1[pt:]
    if random.random() < P_CROSS:
        pt = random.randint(1, LY - 1)
        yb1, yb2 = yb1[:pt] + yb2[pt:], yb2[:pt] + yb1[pt:]
    return (xb1, yb1), (xb2, yb2)


# Мутація
def mutate(ind):
    xb, yb = ind
    def mut(s):
        return ''.join(('1' if c == '0' else '0') if random.random() < P_MUT else c for c in s)
    return mut(xb), mut(yb)


# Селекція методом рулетки
def roulette(pop, cum):
    r = random.random()
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if r <= cum[mid]:
            hi = mid
        else:
            lo = mid + 1
    return pop[lo]


# Головна програма 
def main():
    pop = [(''.join(random.choice('01') for _ in range(LX)),
            ''.join(random.choice('01') for _ in range(LY)))
           for _ in range(POP_SIZE)]

    best_history = []
    avg_history  = []

    for gen in range(GENERATIONS):
        vals = [evaluate(ind)[0] for ind in pop]
        best_idx = max(range(POP_SIZE), key=lambda i: vals[i])
        best_history.append(vals[best_idx])
        avg_history.append(sum(vals) / POP_SIZE)
        fmin = min(vals)
        shifted = [v - fmin + 1e-9 for v in vals]
        total = sum(shifted)
        cum, acc = [], 0.0
        for s in shifted:
            acc += s / total
            cum.append(acc)

        new_pop = [ ] 
        while len(new_pop) < POP_SIZE:
            p1 = roulette(pop, cum)
            p2 = roulette(pop, cum)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POP_SIZE:
                new_pop.append(mutate(c2))
        pop = new_pop

    vals = [evaluate(ind) for ind in pop]
    best_idx = max(range(POP_SIZE), key=lambda i: vals[i][0])
    best_f, best_x, best_y = vals[best_idx]

    print("Координати найкращої точки:")
    print(f"  x = {best_x:.6f}")
    print(f"  y = {best_y:.6f}")
    print(f"Максимум f(x, y) = {best_f:.6f}")

    print("\n# iteration\tbest\tavg")
    for i in range(GENERATIONS):
        print(f"{i}\t{best_history[i]:.6f}\t{avg_history[i]:.6f}")

    # Графік
    plt.figure(figsize=(10, 5))
    plt.plot(range(GENERATIONS), best_history, label="Best",
             color="tab:blue", linewidth=2.0)
    plt.plot(range(GENERATIONS), avg_history,  label="Average",
             color="tab:orange", linewidth=1.2, linestyle="--", alpha=0.85)
    plt.xlabel("Номер ітерації")
    plt.ylabel("Значення f(x, y)")
    plt.title("Збіжність генетичного алгоритму для f(x, y) = 2x + y")
    plt.ylim(10, 14.5)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)
    print("\nГрафік збережено у файл convergence.png")
    plt.show()


if __name__ == "__main__":
    main()