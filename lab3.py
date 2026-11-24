import numpy as np
import matplotlib.pyplot as plt
import sys

class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def sigmoid(u, k):
    return 1.0 / (1.0 + np.exp(-k * u))

def sigmoid_derivative(y, k):
    return y * k * (1.0 - y)

def generate_data(n_points=1200, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.random(n_points)
    x2 = rng.random(n_points)
    d = np.where((x2 < x1 + 0.5) & (x2 > x1 - 0.5), 0.0, 1.0)
    return np.column_stack((x1, x2, d))

def forward(x1, x2, W1, W2, k):
    b = -1.0
    u1 = W1[0, 0] * b + W1[0, 1] * x1 + W1[0, 2] * x2
    u2 = W1[1, 0] * b + W1[1, 1] * x1 + W1[1, 2] * x2
    y1 = sigmoid(u1, k)
    y2 = sigmoid(u2, k)
    u_out = W2[0] * b + W2[1] * y1 + W2[2] * y2
    y = sigmoid(u_out, k)
    return u1, u2, y1, y2, u_out, y

def calculate_test_error(W1, W2, k):
    test_points = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0)]
    error = 0.0
    for x1, x2, d in test_points:
        _, _, _, _, _, y = forward(x1, x2, W1, W2, k)
        error += 0.5 * (d - y) ** 2
    return error

def train_network(data, W1_init, W2_init, eta=1.0, k=1.0, max_epochs=1000):
    W1 = W1_init.copy()
    W2 = W2_init.copy()
    K = len(data)
    
    train_errors = []
    test_errors = []
    best_error = float("inf")
    best_epoch = 0
    best_W1 = W1.copy()
    best_W2 = W2.copy()

    for epoch in range(1, max_epochs + 1):
        total_error = 0.0
        for x1, x2, d in data:
            u1, u2, y1, y2, u_out, y = forward(x1, x2, W1, W2, k)
            eps = d - y
            total_error += 0.5 * eps ** 2

            Fp_out = sigmoid_derivative(y, k)
            Fp_u1 = sigmoid_derivative(y1, k)
            Fp_u2 = sigmoid_derivative(y2, k)

            delta_out = eps * Fp_out
            delta1 = delta_out * W2[1] * Fp_u1
            delta2 = delta_out * W2[2] * Fp_u2

            W2[0] += eta * delta_out * (-1.0)
            W2[1] += eta * delta_out * y1
            W2[2] += eta * delta_out * y2

            W1[0, 0] += eta * delta1 * (-1.0)
            W1[0, 1] += eta * delta1 * x1
            W1[0, 2] += eta * delta1 * x2

            W1[1, 0] += eta * delta2 * (-1.0)
            W1[1, 1] += eta * delta2 * x1
            W1[1, 2] += eta * delta2 * x2

        E_tilde = total_error / K
        E_star = calculate_test_error(W1, W2, k)
        
        train_errors.append(E_tilde)
        test_errors.append(E_star)

        if E_star < best_error:
            best_error = E_star
            best_epoch = epoch
            best_W1 = W1.copy()
            best_W2 = W2.copy()

    return {
        "W1": best_W1,
        "W2": best_W2,
        "train_errors": train_errors,
        "test_errors": test_errors,
        "best_epoch": best_epoch,
        "best_error": best_error
    }

def print_weights_table(W1, W2):
    print(f"{'Нейрон':<30} | {'w0':>10} | {'w1':>10} | {'w2':>10}")
    print("-" * 68)
    print(f"{'Перший нейрон прихованого шару':<30} | {W1[0,0]:>10.6f} | {W1[0,1]:>10.6f} | {W1[0,2]:>10.6f}")
    print(f"{'Другий нейрон прихованого шару':<30} | {W1[1,0]:>10.6f} | {W1[1,1]:>10.6f} | {W1[1,2]:>10.6f}")
    print(f"{'Вихідний нейрон':<30} | {W2[0]:>10.6f} | {W2[1]:>10.6f} | {W2[2]:>10.6f}")

def print_results_table(results, headers):
    print(f"{headers[0]:<20} | {headers[1]:>15} | {headers[2]:>15} | {headers[3]:>15}")
    print("-" * 75)
    for row in results:
        name, epoch, tr_err, te_err = row
        print(f"{name:<20} | {epoch:>15} | {tr_err:>15.6f} | {te_err:>15.10f}")

if __name__ == "__main__":
    sys.stdout = Tee("lab3_output.txt")

    data = generate_data(n_points=1200, seed=42)
    
    W1_START = np.array([[0.1, -0.3, 0.4], [-0.7, -0.1, 0.01]])
    W2_START = np.array([0.4, -0.2, 0.1])

    print("=" * 70)
    res = train_network(data, W1_START, W2_START, eta=1.0, k=1.0, max_epochs=1000)
    
    print(f"\nНайкраща епоха: {res['best_epoch']}")
    print(f"Мінімальна тестова похибка E*: {res['best_error']:.10f}")
    print("\nНайкращі вагові коефіцієнти (на момент найкращої епохи):")
    print_weights_table(res["W1"], res["W2"])

    print("\nПеревірка XOR (найкращі ваги):")
    test_pts = [(0.0, 0.0, 0), (1.0, 0.0, 1), (0.0, 1.0, 1), (1.0, 1.0, 0)]
    print(f"{'x1':>4} | {'x2':>4} | {'Очікувано':>10} | {'Отримано':>10} | {'Клас':>5}")
    print("-" * 45)
    for x1, x2, d in test_pts:
        _, _, _, _, _, y = forward(x1, x2, res["W1"], res["W2"], 1.0)
        pred_class = int(y >= 0.5)
        print(f"{x1:>4.0f} | {x2:>4.0f} | {d:>10} | {y:>10.4f} | {pred_class:>5}")

    print("=" * 70)
    print("\n")
    np.random.seed(42)
    W1_RAND = np.random.uniform(-0.5, 0.5, size=(2, 3))
    W2_RAND = np.random.uniform(-0.5, 0.5, size=3)
    W1_OTHER = np.array([[0.5, 0.2, -0.4], [-0.3, 0.6, 0.1]])
    W2_OTHER = np.array([0.2, 0.3, -0.5])
    
    weights_variants = [
        ("Рекомендовані", W1_START, W2_START),
        ("Інші", W1_OTHER, W2_OTHER),
        ("Випадкові", W1_RAND, W2_RAND),
    ]
    
    res_weights = []
    for name, W1_v, W2_v in weights_variants:
        r = train_network(data, W1_v, W2_v, eta=1.0, k=1.0, max_epochs=1000)
        idx = r["best_epoch"] - 1
        tr_err = r["train_errors"][idx]
        res_weights.append((name, r["best_epoch"], tr_err, r["best_error"]))
    print_results_table(res_weights, ["Початкові ваги", "Найкраща епоха", "Навчальна похибка", "Тестова похибка E*"])

    print("=" * 70)
    print("\n")
    eta_values = [0.01, 0.1, 0.5, 1.0]
    res_eta = []
    for eta_v in eta_values:
        r = train_network(data, W1_START, W2_START, eta=eta_v, k=1.0, max_epochs=1000)
        idx = r["best_epoch"] - 1
        tr_err = r["train_errors"][idx]
        res_eta.append((f"{eta_v:.2f}", r["best_epoch"], tr_err, r["best_error"]))
    print_results_table(res_eta, ["eta", "Найкраща епоха", "Навчальна похибка", "Тестова похибка E*"])

    print("=" * 70)
    print("\n")
    k_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    res_k = []
    for k_v in k_values:
        r = train_network(data, W1_START, W2_START, eta=1.0, k=k_v, max_epochs=1000)
        idx = r["best_epoch"] - 1
        tr_err = r["train_errors"][idx]
        res_k.append((f"{k_v:.1f}", r["best_epoch"], tr_err, r["best_error"]))
    print_results_table(res_k, ["k", "Найкраща епоха", "Навчальна похибка", "Тестова похибка E*"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, len(res["test_errors"]) + 1), res["test_errors"], label="Тестова похибка E*", color="blue")
    axes[0].axvline(res["best_epoch"], color="red", linestyle="--", label=f"Найкраща епоха: {res['best_epoch']}")
    axes[0].set_xlabel("Кількість епох")
    axes[0].set_ylabel("Тестова похибка E*")
    axes[0].set_title("Залежність тестової похибки E* від кількості епох")
    axes[0].grid(True)
    axes[0].legend()

    for k_v in k_values:
        r_k = train_network(data, W1_START, W2_START, eta=1.0, k=k_v, max_epochs=1000)
        axes[1].plot(range(1, len(r_k["test_errors"]) + 1), r_k["test_errors"], label=f"k = {k_v}")
    
    axes[1].set_xlabel("Кількість епох")
    axes[1].set_ylabel("Тестова похибка E*")
    axes[1].set_title("Залежність E* від епох для різних k")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("all_plots.png", dpi=150)
    plt.show()
    
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal