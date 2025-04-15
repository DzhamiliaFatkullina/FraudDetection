import joblib
import random
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# 1) Load pretrained pipelines
d = joblib.load('models.joblib')
base_rf_pipe  = d['RandomForest']
base_gb_pipe  = d['GradientBoosting']

# 2) Load data splits
ds = joblib.load('train_test_data.joblib')
X_train, X_test = ds['X_train'], ds['X_test']
y_train, y_test = ds['y_train'], ds['y_test']

# 3) Caches for fitness to avoid recomputation
_cache_rf  = {}
_cache_gb  = {}

def _make_key(params):
    # round floats to 6 decimals for caching stability
    return tuple(np.round(params, 6))

# 4) Fitness functions with caching


def fitness_rf(params):
    key = _make_key(params)
    if key in _cache_rf:
        return _cache_rf[key]
    ne, md = params
    ne = max(1, int(round(ne)))
    md = None if md < 1 else int(round(md))
    pipe = clone(base_rf_pipe)
    pipe.set_params(model__n_estimators=ne, model__max_depth=md)
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_prob)
    _cache_rf[key] = score
    return score


def fitness_gb(params):
    key = _make_key(params)
    if key in _cache_gb:
        return _cache_gb[key]
    lr, ne = params
    ne = max(1, int(round(ne)))
    pipe = clone(base_gb_pipe)
    pipe.set_params(model__learning_rate=lr, model__n_estimators=ne)
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_prob)
    _cache_gb[key] = score
    return score

# 5) Early_stopping threshold
NO_IMPROVE_LIMIT = 3

# 6) Genetic Algorithm with early stopping
def genetic_algorithm(fitness_func, population_size=5, generations=10, mutation_rate=0.2):
    population = [np.random.uniform(1, 100, size=2) for _ in range(population_size)]
    history = []
    best_score = -np.inf
    no_improve = 0
    for gen in range(generations):
        scores = [fitness_func(ind) for ind in population]
        current_best = max(scores)
        history.append(current_best)
        if current_best > best_score:
            best_score = current_best
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= NO_IMPROVE_LIMIT:
                print(f"GA early stop at gen {gen}, no improvement for {NO_IMPROVE_LIMIT} gens")
                break
        # selection and reproduction
        idx_sorted = np.argsort(scores)[::-1]
        selected = [population[i] for i in idx_sorted[:population_size // 2]]
        children = []
        while len(children) < population_size:
            p1, p2 = random.sample(selected, 2)
            child = np.array([p1[0] if random.random() < 0.5 else p2[0],
                              p1[1] if random.random() < 0.5 else p2[1]])
            if random.random() < mutation_rate:
                child += np.random.uniform(-1, 1, size=2)
            children.append(np.clip(child, 1, 100))
        population = children
    # final best
    final_scores = [fitness_func(ind) for ind in population]
    best_idx = int(np.argmax(final_scores))
    return population[best_idx], history

# 7) Particle Swarm Optimization with early stopping
def particle_swarm_optimization(fitness_func, swarm_size=5, iterations=10):
    particles = [np.random.uniform(1, 100, size=2) for _ in range(swarm_size)]
    velocities = [np.zeros(2) for _ in range(swarm_size)]
    personal_best = particles.copy()
    global_best = particles[0]
    global_score = fitness_func(global_best)
    history = []
    no_improve = 0
    for it in range(iterations):
        scores = [fitness_func(p) for p in particles]
        current_best = max(scores)
        history.append(current_best)
        if current_best > global_score:
            global_score = current_best
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= NO_IMPROVE_LIMIT:
                print(f"PSO early stop at iter {it}, no improvement for {NO_IMPROVE_LIMIT} iters")
                break
        for i in range(len(particles)):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (0.5 * velocities[i]
                             + r1 * (personal_best[i] - particles[i])
                             + r2 * (global_best - particles[i]))
            particles[i] = np.clip(particles[i] + velocities[i], 1, 100)
            # update personal best
            score = fitness_func(particles[i])
            if score > fitness_func(personal_best[i]):
                personal_best[i] = particles[i].copy()
        if fitness_func(global_best) < current_best:
            global_best = particles[int(np.argmax(scores))].copy()
    return global_best, history

# 8) Artificial Immune System with early stopping
def artificial_immune_system(fitness_func, population_size=5, generations=10, clone_factor=5):
    antibodies = [np.random.uniform(1, 100, size=2) for _ in range(population_size)]
    history = []
    best_score = -np.inf
    no_improve = 0
    for gen in range(generations):
        pool = []
        for ab in antibodies:
            clones = [np.clip(ab + np.random.normal(0, 5, size=2), 1, 100)
                      for _ in range(clone_factor)]
            pool.extend(clones)
        scores = [fitness_func(cl) for cl in pool]
        current_best = max(scores)
        history.append(current_best)
        if current_best > best_score:
            best_score = current_best
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= NO_IMPROVE_LIMIT:
                print(f"AIS early stop at gen {gen}, no improvement for {NO_IMPROVE_LIMIT} gens")
                break
        # select top antibodies
        idx_sorted = np.argsort(scores)[-population_size:]
        antibodies = [pool[i] for i in idx_sorted]
    final_scores = [fitness_func(ab) for ab in antibodies]
    best_ab = antibodies[int(np.argmax(final_scores))]
    return best_ab, history
