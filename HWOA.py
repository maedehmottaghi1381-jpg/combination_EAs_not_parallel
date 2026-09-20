"""
Hybrid Whale-Turtle-Genetic Algorithm (HWTOA) for Feature Selection
Dataset: Heart Disease (UCI)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import urllib.request
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ============================================================
# 1. Load Heart Disease dataset
# ============================================================
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

print("=" * 60)
print("Loading Heart Disease dataset")
print("=" * 60)

try:
    print("Downloading from UCI...")
    with urllib.request.urlopen(URL, timeout=30) as response:
        raw = response.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(raw), names=COLUMNS, na_values='?')
    print("Download successful.")
except Exception as e:
    print(f"Download failed: {e}")
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='heart-disease', version=1, as_frame=True, parser='auto')
    df = data.frame
    df.columns = [c.lower() for c in df.columns]

print(f"Dataset shape: {df.shape}")

# ---------- Prepare X and y ----------
if 'target' in df.columns:
    y_raw = df['target']
    X_df = df.drop(columns=['target'])
else:
    y_raw = df.iloc[:, -1]
    X_df = df.iloc[:, :-1]

# Convert target to numeric and binarize (0 = healthy, >0 = diseased)
y_numeric = pd.to_numeric(y_raw, errors='coerce').fillna(0)
y_binary = (y_numeric > 0).astype(int).values

# Impute and scale
X_imputed = SimpleImputer(strategy='median').fit_transform(X_df)
X_scaled = StandardScaler().fit_transform(X_imputed)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Number of features: {X_train.shape[1]}")


# ============================================================
# 2. Fitness function
# ============================================================
def fitness_function(selected_features, X_train, y_train, X_test, y_test,
                     alpha=0.99, beta=0.01):
    if np.sum(selected_features) == 0:
        return -1.0
    try:
        clf = SVC(kernel='rbf', C=1.0, random_state=42)
        clf.fit(X_train[:, selected_features], y_train)
        acc = accuracy_score(y_test, clf.predict(X_test[:, selected_features]))
    except Exception:
        return -1.0
    return alpha * acc - beta * (np.sum(selected_features) / X_train.shape[1])


# ============================================================
# 3. WOA
# ============================================================
def whale_optimization_algorithm(X_train, y_train, X_test, y_test,
                                 n_whales=20, max_iter=50):
    n_features = X_train.shape[1]
    positions = np.random.rand(n_whales, n_features)
    fitness_values = np.array([fitness_function(p > 0.5, X_train, y_train, X_test, y_test) for p in positions])

    best_idx = np.argmax(fitness_values)
    best_position = positions[best_idx].copy()
    best_fitness = fitness_values[best_idx]
    convergence = [best_fitness]

    for t in range(max_iter):
        a = 2 - 2 * t / max_iter
        for i in range(n_whales):
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()
            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * best_position - positions[i])
                    positions[i] = best_position - A * D
                else:
                    rand_idx = np.random.randint(n_whales)
                    D = np.abs(C * positions[rand_idx] - positions[i])
                    positions[i] = positions[rand_idx] - A * D
            else:
                D = np.abs(best_position - positions[i])
                l = np.random.uniform(-1, 1)
                positions[i] = D * np.exp(l) * np.cos(2 * np.pi * l) + best_position
            positions[i] = np.clip(positions[i], 0, 1)
            fit = fitness_function(positions[i] > 0.5, X_train, y_train, X_test, y_test)
            if fit > best_fitness:
                best_fitness, best_position = fit, positions[i].copy()
        convergence.append(best_fitness)
    return best_position > 0.5, best_fitness, convergence


# ============================================================
# 4. TOA
# ============================================================
def turtle_optimization_algorithm(X_train, y_train, X_test, y_test,
                                  n_turtles=20, max_iter=50):
    n_features = X_train.shape[1]
    positions = np.random.rand(n_turtles, n_features)
    fitness_values = np.array([fitness_function(p > 0.5, X_train, y_train, X_test, y_test) for p in positions])

    best_idx = np.argmax(fitness_values)
    best_position = positions[best_idx].copy()
    best_fitness = fitness_values[best_idx]
    convergence = [best_fitness]

    for t in range(max_iter):
        geomagnetic = np.sin(2 * np.pi * t / max_iter)
        ocean_current = np.random.randn(n_turtles, n_features) * 0.1
        sorted_idx = np.argsort(fitness_values)[::-1]
        reference = positions[sorted_idx[np.random.randint(0, min(5, n_turtles))]]
        for i in range(n_turtles):
            direction = reference - positions[i]
            step = geomagnetic * direction + ocean_current[i]
            stamina = 1.0 - t / max_iter
            step *= (0.5 + 0.5 * stamina)
            positions[i] = np.clip(positions[i] + step, 0, 1)
            fit = fitness_function(positions[i] > 0.5, X_train, y_train, X_test, y_test)
            if fit > best_fitness:
                best_fitness, best_position = fit, positions[i].copy()
        fitness_values = np.array([fitness_function(p > 0.5, X_train, y_train, X_test, y_test) for p in positions])
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_position = positions[current_best_idx].copy()
        convergence.append(best_fitness)
    return best_position > 0.5, best_fitness, convergence


# ============================================================
# 5. GA
# ============================================================
def genetic_algorithm(X_train, y_train, X_test, y_test,
                      population_size=20, generations=50,
                      crossover_rate=0.8, mutation_rate=0.1):
    n_features = X_train.shape[1]
    population = np.random.randint(0, 2, size=(population_size, n_features))
    fitness_values = np.array([fitness_function(ind.astype(bool), X_train, y_train, X_test, y_test) for ind in population])

    best_idx = np.argmax(fitness_values)
    best_individual = population[best_idx].copy()
    best_fitness = fitness_values[best_idx]
    convergence = [best_fitness]

    for gen in range(generations):
        new_population = []
        for _ in range(population_size):
            idx = np.random.choice(population_size, 3, replace=False)
            winner = idx[np.argmax(fitness_values[idx])]
            new_population.append(population[winner].copy())
        new_population = np.array(new_population)

        for i in range(0, population_size - 1, 2):
            if np.random.rand() < crossover_rate:
                cp = np.random.randint(1, n_features)
                temp = new_population[i, cp:].copy()
                new_population[i, cp:] = new_population[i + 1, cp:]
                new_population[i + 1, cp:] = temp

        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mp = np.random.randint(0, n_features)
                new_population[i, mp] = 1 - new_population[i, mp]

        population = new_population
        fitness_values = np.array([fitness_function(ind.astype(bool), X_train, y_train, X_test, y_test) for ind in population])
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_individual = population[current_best_idx].copy()
        convergence.append(best_fitness)
    return best_individual.astype(bool), best_fitness, convergence


# ============================================================
# 6. HWTOA (Hybrid)
# ============================================================
def hybrid_whale_turtle_genetic_algorithm(X_train, y_train, X_test, y_test,
                                           n_agents=20, max_iter=50,
                                           crossover_rate=0.8, mutation_rate=0.1):
    n_features = X_train.shape[1]
    population = np.random.rand(n_agents, n_features)
    fitness_values = np.array([fitness_function(ind > 0.5, X_train, y_train, X_test, y_test) for ind in population])

    best_idx = np.argmax(fitness_values)
    best_individual = population[best_idx].copy()
    best_fitness = fitness_values[best_idx]
    convergence = [best_fitness]

    phase1_end = max_iter // 3
    phase2_end = 2 * max_iter // 3

    for t in range(max_iter):
        if t < phase1_end:
            a = 2 - 2 * t / phase1_end
            for i in range(n_agents):
                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = np.random.rand()
                if p < 0.5:
                    if abs(A) < 1:
                        D = np.abs(C * best_individual - population[i])
                        population[i] = best_individual - A * D
                    else:
                        rand_idx = np.random.randint(n_agents)
                        D = np.abs(C * population[rand_idx] - population[i])
                        population[i] = population[rand_idx] - A * D
                else:
                    D = np.abs(best_individual - population[i])
                    l = np.random.uniform(-1, 1)
                    population[i] = D * np.exp(l) * np.cos(2 * np.pi * l) + best_individual
                population[i] = np.clip(population[i], 0, 1)

        elif t < phase2_end:
            new_pop = []
            for _ in range(n_agents):
                idx = np.random.choice(n_agents, 3, replace=False)
                winner = idx[np.argmax(fitness_values[idx])]
                new_pop.append(population[winner].copy())
            new_pop = np.array(new_pop)
            for i in range(0, n_agents - 1, 2):
                if np.random.rand() < crossover_rate:
                    cp = np.random.randint(1, n_features)
                    temp = new_pop[i, cp:].copy()
                    new_pop[i, cp:] = new_pop[i + 1, cp:]
                    new_pop[i + 1, cp:] = temp
            for i in range(n_agents):
                if np.random.rand() < mutation_rate:
                    mp = np.random.randint(0, n_features)
                    new_pop[i, mp] = 1 - new_pop[i, mp]
            for i in range(n_agents):
                if np.random.rand() < 0.3:
                    alpha = np.random.rand()
                    new_pop[i] = alpha * new_pop[i] + (1 - alpha) * best_individual
                    new_pop[i] = np.clip(new_pop[i], 0, 1)
            population = new_pop

        else:
            geomagnetic = np.sin(2 * np.pi * t / max_iter)
            ocean_current = np.random.randn(n_agents, n_features) * 0.05
            for i in range(n_agents):
                direction = best_individual - population[i]
                step = geomagnetic * direction + ocean_current[i]
                stamina = 1.0 - t / max_iter
                step *= (0.3 + 0.7 * stamina)
                population[i] = np.clip(population[i] + step, 0, 1)

        fitness_values = np.array([fitness_function(ind > 0.5, X_train, y_train, X_test, y_test) for ind in population])
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_individual = population[current_best_idx].copy()
        convergence.append(best_fitness)
        if (t + 1) % 10 == 0:
            print(f"  HWTOA - iter {t + 1}/{max_iter}, best fitness: {best_fitness:.4f}")
    return best_individual > 0.5, best_fitness, convergence


# ============================================================
# 7. Evaluate
# ============================================================
def evaluate_algorithm(selected_mask, X_train, y_train, X_test, y_test):
    if np.sum(selected_mask) == 0:
        return 0.0, 0
    clf = SVC(kernel='rbf', C=1.0, random_state=42)
    clf.fit(X_train[:, selected_mask], y_train)
    acc = accuracy_score(y_test, clf.predict(X_test[:, selected_mask]))
    return acc, int(np.sum(selected_mask))


# ============================================================
# 8. Run all
# ============================================================
print("\n" + "=" * 60)
print("Running algorithms")
print("=" * 60)

print("\n[1/4] WOA...")
woa_mask, woa_fit, woa_conv = whale_optimization_algorithm(X_train, y_train, X_test, y_test)
woa_acc, woa_n = evaluate_algorithm(woa_mask, X_train, y_train, X_test, y_test)
print(f"WOA   - Accuracy: {woa_acc:.4f} | Features: {woa_n}/{X_train.shape[1]}")

print("\n[2/4] TOA...")
toa_mask, toa_fit, toa_conv = turtle_optimization_algorithm(X_train, y_train, X_test, y_test)
toa_acc, toa_n = evaluate_algorithm(toa_mask, X_train, y_train, X_test, y_test)
print(f"TOA   - Accuracy: {toa_acc:.4f} | Features: {toa_n}/{X_train.shape[1]}")

print("\n[3/4] GA...")
ga_mask, ga_fit, ga_conv = genetic_algorithm(X_train, y_train, X_test, y_test)
ga_acc, ga_n = evaluate_algorithm(ga_mask, X_train, y_train, X_test, y_test)
print(f"GA    - Accuracy: {ga_acc:.4f} | Features: {ga_n}/{X_train.shape[1]}")

print("\n[4/4] HWTOA...")
hwtoa_mask, hwtoa_fit, hwtoa_conv = hybrid_whale_turtle_genetic_algorithm(X_train, y_train, X_test, y_test)
hwtoa_acc, hwtoa_n = evaluate_algorithm(hwtoa_mask, X_train, y_train, X_test, y_test)
print(f"HWTOA - Accuracy: {hwtoa_acc:.4f} | Features: {hwtoa_n}/{X_train.shape[1]}")


# ============================================================
# 9. Final comparison
# ============================================================
print("\n" + "=" * 60)
print("Final Comparison")
print("=" * 60)

results = pd.DataFrame({
    'Algorithm': ['WOA', 'TOA', 'GA', 'HWTOA'],
    'Accuracy': [woa_acc, toa_acc, ga_acc, hwtoa_acc],
    'Selected Features': [woa_n, toa_n, ga_n, hwtoa_n],
    'Best Fitness': [woa_fit, toa_fit, ga_fit, hwtoa_fit]
}).sort_values('Accuracy', ascending=False).reset_index(drop=True)

results.index = results.index + 1
print("\n" + results.to_string())
print("\nDone!")