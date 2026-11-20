import math
import matplotlib.pyplot as plt

dt = 0.1                
T = 200.0                 
d_mu = 0.1               
mu_min = -math.pi / 36   
mu_max = math.pi / 36    
Chislo_pravil = 36       
phi_min, phi_max = -math.pi, math.pi
omega_min, omega_max = -math.pi / 9, math.pi / 9

# Варіант 16
phi_deg = -50.0
omega_deg = -1.0

phi = math.radians(phi_deg)
omega = math.radians(omega_deg)

# Функція належності
def F(x, a, b, c, d):
    if x <= a: return 0.0
    elif a <= x <= b: return (x - a) / (b - a)
    elif b <= x <= c: return 1.0
    elif c <= x <= d: return (d - x) / (d - c)
    else: return 0.0

terms = [
    (-1000, -200, -100, -50),
    (-100, -50, -50, -10),
    (-50, -10, 0, 0),
    (0, 0, 10, 50),
    (10, 50, 50, 100),
    (50, 100, 200, 1000)
]

rules = [
    (1,1,6), (1,2,6), (1,3,6), (1,4,6), (1,5,6), (1,6,6),
    (2,1,6), (2,2,6), (2,3,6), (2,4,5), (2,5,5), (2,6,5),
    (3,1,6), (3,2,6), (3,3,6), (3,4,5), (3,5,5), (3,6,5),
    (4,1,2), (4,2,2), (4,3,2), (4,4,1), (4,5,1), (4,6,1),
    (5,1,2), (5,2,2), (5,3,2), (5,4,1), (5,5,1), (5,6,1),
    (6,1,1), (6,2,1), (6,3,1), (6,4,1), (6,5,1), (6,6,1)
]

# Алгоритм визначення загальної ФП
def get_chi_star(phi_norm, omega_norm, mu_norm):
    chi_star = 0.0
    for n1, n2, m in rules:
        a_phi = F(phi_norm, *terms[n1-1])
        b_omega = F(omega_norm, *terms[n2-1])
        gamma = min(a_phi, b_omega)
        delta = F(mu_norm, *terms[m-1])
        chi = min(gamma, delta)
        if chi > chi_star:
            chi_star = chi    
    return chi_star

time_data = []
phi_data = []
omega_data = []
mu_data = []

t = 0.0
while t < T - 0.5 * dt:
    # Приведення до безрозмірного діапазону [-100, 100]
    phi_norm = (200.0 / (phi_max - phi_min)) * (phi - phi_min) - 100.0
    omega_norm = (200.0 / (omega_max - omega_min)) * (omega - omega_min) - 100.0
    mu_norm = -100.0
    s1 = 0.0
    s2 = 0.0

    while mu_norm < 100.0 - 0.5 * d_mu:
        chi_star = get_chi_star(phi_norm, omega_norm, mu_norm)
        s1 += mu_norm * chi_star * d_mu
        s2 += chi_star * d_mu
        mu_norm += d_mu

    #Дефазіфікація
    if s2 == 0:
        mu_star = 0.0
    else:
        mu_star = s1 / s2

    #Зворотне перетворення
    mu = ((mu_star + 100.0) / 200.0) * (mu_max - mu_min) + mu_min

    time_data.append(t)
    phi_data.append(math.degrees(phi))
    omega_data.append(math.degrees(omega))
    mu_data.append(math.degrees(mu))

    # Метод Ейлера
    phi += omega * dt
    omega += mu * dt
    t += dt

with open("result.txt", "w", encoding="utf-8") as f:
    f.write("t(s)\tphi(deg)\tomega(deg/s)\t mu(deg/s^2)\n")
    for i in range(len(time_data)):
        f.write(f"{time_data[i]:.2f}\t{phi_data[i]:.4f}\t{omega_data[i]:.4f}\t {mu_data[i]:.4f}\n")

print("Моделювання завершено. Дані збережено у result.txt")

plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(time_data, phi_data, color='red')
plt.ylabel('Кут φ (град.)')
plt.grid(True)
plt.subplot(3, 1, 2)
plt.plot(time_data, omega_data, color='blue')
plt.ylabel('Швидкість ω (град./с)')
plt.grid(True)
plt.subplot(3, 1, 3)
plt.plot(time_data, mu_data, color='green')
plt.ylabel('Прискорення μ (град./с²)')
plt.xlabel('Час t (с)')
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs.png", dpi=150)
plt.show()