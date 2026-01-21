#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 3: The Strategic Wait - Model B: Multi-Objective Optimization (NSGA-II)
Dynamic Parking Strategy Optimization using NSGA-II algorithm
Optimizes three objectives: AWT, Long Wait Ratio, Energy Consumption
All outputs are in English.
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import random

# Output directory
OUTPUT_DIR = "output/model_B"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data directory
PREPROCESS_DIR = "output/preprocessed"


class ParkingStrategy:
    """Represents a parking strategy as an individual in NSGA-II"""
    
    def __init__(self, num_elevators, min_floor, max_floor, genes=None):
        self.num_elevators = num_elevators
        self.min_floor = min_floor
        self.max_floor = max_floor
        
        if genes is None:
            # Random initialization with diverse strategies
            self.genes = self._random_init()
        else:
            self.genes = genes.copy()
        
        # Objectives (to be calculated)
        self.objectives = None  # [AWT, long_wait_ratio, energy]
        
        # NSGA-II attributes
        self.rank = None
        self.crowding_distance = 0
    
    def _random_init(self):
        """Generate random parking configuration with diversity"""
        genes = []
        
        # Choose a random strategy type
        strategy_type = random.choice(['uniform', 'clustered', 'demand_based', 'random'])
        
        if strategy_type == 'uniform':
            # Uniform distribution across floors
            step = (self.max_floor - self.min_floor) / max(1, self.num_elevators - 1)
            for i in range(self.num_elevators):
                floor = self.min_floor + int(i * step)
                floor = max(self.min_floor, min(self.max_floor, floor))
                # Add small random perturbation
                floor += random.randint(-1, 1)
                floor = max(self.min_floor, min(self.max_floor, floor))
                genes.append(floor)
        
        elif strategy_type == 'clustered':
            # Cluster around a random center
            center = random.randint(self.min_floor, self.max_floor)
            spread = random.randint(1, 3)
            for i in range(self.num_elevators):
                floor = center + random.randint(-spread, spread)
                floor = max(self.min_floor, min(self.max_floor, floor))
                genes.append(floor)
        
        elif strategy_type == 'demand_based':
            # Bias towards lower floors (typically higher demand)
            for i in range(self.num_elevators):
                # Beta distribution biased towards lower floors
                ratio = np.random.beta(2, 5)
                floor = int(self.min_floor + ratio * (self.max_floor - self.min_floor))
                genes.append(floor)
        
        else:  # random
            for i in range(self.num_elevators):
                genes.append(random.randint(self.min_floor, self.max_floor))
        
        return genes
    
    def get_parking_floors(self):
        """Get parking floor configuration"""
        return self.genes
    
    def mutate(self, mutation_rate=0.2, mutation_strength=3):
        """Mutate genes with variable strength"""
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                # Choose mutation type
                if random.random() < 0.5:
                    # Small local mutation
                    delta = random.randint(-mutation_strength, mutation_strength)
                    self.genes[i] += delta
                else:
                    # Large jump mutation for exploration
                    self.genes[i] = random.randint(self.min_floor, self.max_floor)
                
                # Clamp to valid range
                self.genes[i] = max(self.min_floor, min(self.max_floor, self.genes[i]))
    
    def crossover(self, other):
        """Crossover with another individual"""
        child1_genes = []
        child2_genes = []
        
        # Multi-point crossover
        crossover_points = sorted(random.sample(range(self.num_elevators), 
                                                min(2, self.num_elevators)))
        
        use_self = True
        point_idx = 0
        
        for i in range(self.num_elevators):
            if point_idx < len(crossover_points) and i == crossover_points[point_idx]:
                use_self = not use_self
                point_idx += 1
            
            if use_self:
                child1_genes.append(self.genes[i])
                child2_genes.append(other.genes[i])
            else:
                child1_genes.append(other.genes[i])
                child2_genes.append(self.genes[i])
        
        child1 = ParkingStrategy(self.num_elevators, self.min_floor, self.max_floor, child1_genes)
        child2 = ParkingStrategy(self.num_elevators, self.min_floor, self.max_floor, child2_genes)
        
        return child1, child2


class ObjectiveCalculator:
    """Calculate objectives for parking strategies"""
    
    def __init__(self, params, demand_data, wait_data=None):
        self.params = params
        self.demand_data = demand_data
        self.wait_data = wait_data
        
        self.num_floors = params['max_floor'] - params['min_floor'] + 1
        self.min_floor = params['min_floor']
        self.max_floor = params['max_floor']
        
        # Create floor demand weights
        self.floor_weights = self._calculate_floor_weights()
        
        # Physical parameters
        self.travel_time_per_floor = params.get('avg_travel_time_per_floor', 3.0)
        self.door_time = params.get('door_open_time', 4.0)
    
    def _calculate_floor_weights(self):
        """Calculate demand weights for each floor"""
        weights = {}
        
        if self.demand_data is not None and len(self.demand_data) > 0:
            total = self.demand_data['call_count'].sum()
            for _, row in self.demand_data.iterrows():
                floor = int(row['floor'])
                weights[floor] = row['call_count'] / total if total > 0 else 1.0 / self.num_floors
        else:
            # Uniform weights
            for floor in range(self.min_floor, self.max_floor + 1):
                weights[floor] = 1.0 / self.num_floors
        
        return weights
    
    def calculate_objectives(self, strategy):
        """
        Calculate three objectives:
        1. Average Wait Time (AWT) - minimize
        2. Long Wait Ratio (>60s) - minimize  
        3. Energy Consumption (movement cost) - minimize
        """
        parking_floors = strategy.get_parking_floors()
        
        # Objective 1: Average Wait Time
        awt = self._calculate_awt(parking_floors)
        
        # Objective 2: Long Wait Ratio
        long_wait_ratio = self._calculate_long_wait_ratio(parking_floors)
        
        # Objective 3: Energy Consumption
        energy = self._calculate_energy(parking_floors)
        
        return [awt, long_wait_ratio, energy]
    
    def _calculate_awt(self, parking_floors):
        """Calculate expected average wait time"""
        total_wait = 0
        total_weight = 0
        
        for floor in range(self.min_floor, self.max_floor + 1):
            weight = self.floor_weights.get(floor, 0)
            
            # Find nearest elevator
            min_distance = min(abs(floor - pf) for pf in parking_floors)
            
            # Wait time = travel time + door time + random wait component
            travel_time = min_distance * self.travel_time_per_floor
            wait_time = travel_time + self.door_time
            
            # Add small random component for realistic variation
            wait_time += np.random.normal(0, 2)
            wait_time = max(0, wait_time)
            
            total_wait += weight * wait_time
            total_weight += weight
        
        awt = total_wait / total_weight if total_weight > 0 else 30.0
        return awt
    
    def _calculate_long_wait_ratio(self, parking_floors):
        """Calculate ratio of floors with expected wait > 60s"""
        threshold = 60.0
        long_wait_count = 0
        total_weight = 0
        
        for floor in range(self.min_floor, self.max_floor + 1):
            weight = self.floor_weights.get(floor, 0)
            
            # Calculate expected wait
            min_distance = min(abs(floor - pf) for pf in parking_floors)
            wait_time = min_distance * self.travel_time_per_floor + self.door_time
            
            # In congested scenarios, wait time can be much higher
            # Add congestion factor based on distance from ground floor
            congestion_factor = 1.0 + 0.05 * abs(floor - self.min_floor)
            wait_time *= congestion_factor
            
            if wait_time > threshold:
                long_wait_count += weight
            
            total_weight += weight
        
        ratio = long_wait_count / total_weight if total_weight > 0 else 0.1
        return ratio * 100  # Convert to percentage
    
    def _calculate_energy(self, parking_floors):
        """Calculate energy consumption proxy (total repositioning distance)"""
        # Energy is proportional to expected travel distance
        total_energy = 0
        
        for floor in range(self.min_floor, self.max_floor + 1):
            weight = self.floor_weights.get(floor, 0)
            
            # Find nearest elevator
            min_distance = min(abs(floor - pf) for pf in parking_floors)
            
            # Energy cost increases with distance
            energy_cost = min_distance * 1.5  # Units of energy per floor
            total_energy += weight * energy_cost
        
        # Add repositioning cost (moving elevators to parking positions)
        # Assume starting from ground floor
        for pf in parking_floors:
            total_energy += abs(pf - self.min_floor) * 0.5
        
        return total_energy


class NSGA2Optimizer:
    """NSGA-II algorithm for multi-objective optimization"""
    
    def __init__(self, params, objective_calculator, population_size=100):
        self.params = params
        self.calculator = objective_calculator
        self.population_size = population_size
        
        self.num_elevators = params['num_elevators']
        self.min_floor = params['min_floor']
        self.max_floor = params['max_floor']
        
        # Initialize population
        self.population = self._init_population()
        
        # History
        self.history = {
            'generations': [],
            'best_awt': [],
            'avg_awt': [],
            'pareto_size': []
        }
        
        print(f"NSGA-II initialized:")
        print(f"  - Population size: {population_size}")
        print(f"  - Elevators: {self.num_elevators}")
        print(f"  - Floors: {self.min_floor}-{self.max_floor}")
    
    def _init_population(self):
        """Initialize diverse population"""
        population = []
        
        # Add some predefined strategies for diversity
        predefined = [
            [self.min_floor] * self.num_elevators,  # All at ground
            [self.max_floor] * self.num_elevators,  # All at top
            list(range(self.min_floor, self.min_floor + self.num_elevators)),  # Sequential
        ]
        
        for genes in predefined:
            if len(genes) >= self.num_elevators:
                ind = ParkingStrategy(self.num_elevators, self.min_floor, 
                                     self.max_floor, genes[:self.num_elevators])
                population.append(ind)
        
        # Fill rest with random individuals
        while len(population) < self.population_size:
            ind = ParkingStrategy(self.num_elevators, self.min_floor, self.max_floor)
            population.append(ind)
        
        return population
    
    def evaluate_population(self, population):
        """Evaluate objectives for all individuals"""
        for ind in population:
            if ind.objectives is None:
                ind.objectives = self.calculator.calculate_objectives(ind)
    
    def dominates(self, ind1, ind2):
        """Check if ind1 dominates ind2 (all objectives less or equal, at least one strictly less)"""
        better_in_any = False
        for o1, o2 in zip(ind1.objectives, ind2.objectives):
            if o1 > o2:  # Worse in this objective
                return False
            if o1 < o2:
                better_in_any = True
        return better_in_any
    
    def non_dominated_sort(self, population):
        """Fast non-dominated sorting"""
        fronts = [[]]
        
        # Calculate domination for each individual
        domination_count = {id(ind): 0 for ind in population}
        dominated_by = {id(ind): [] for ind in population}
        
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i != j:
                    if self.dominates(p, q):
                        dominated_by[id(p)].append(q)
                    elif self.dominates(q, p):
                        domination_count[id(p)] += 1
            
            if domination_count[id(p)] == 0:
                p.rank = 0
                fronts[0].append(p)
        
        # Generate subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for p in fronts[current_front]:
                for q in dominated_by[id(p)]:
                    domination_count[id(q)] -= 1
                    if domination_count[id(q)] == 0:
                        q.rank = current_front + 1
                        next_front.append(q)
            current_front += 1
            fronts.append(next_front)
        
        # Remove empty last front
        if not fronts[-1]:
            fronts.pop()
        
        return fronts
    
    def calculate_crowding_distance(self, front):
        """Calculate crowding distance for individuals in a front"""
        n = len(front)
        if n <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        num_objectives = len(front[0].objectives)
        
        for ind in front:
            ind.crowding_distance = 0
        
        for m in range(num_objectives):
            # Sort by this objective
            front.sort(key=lambda x: x.objectives[m])
            
            # Boundary points get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0:
                continue
            
            # Calculate distances
            for i in range(1, n - 1):
                distance = (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range
                front[i].crowding_distance += distance
    
    def tournament_selection(self, population, tournament_size=2):
        """Binary tournament selection based on rank and crowding distance"""
        candidates = random.sample(population, tournament_size)
        
        # Select best by rank first, then crowding distance
        best = candidates[0]
        for c in candidates[1:]:
            if c.rank < best.rank:
                best = c
            elif c.rank == best.rank and c.crowding_distance > best.crowding_distance:
                best = c
        
        return best
    
    def create_offspring(self, population):
        """Create offspring population through selection, crossover, and mutation"""
        offspring = []
        
        while len(offspring) < self.population_size:
            # Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            if random.random() < 0.9:  # High crossover rate
                child1, child2 = parent1.crossover(parent2)
            else:
                child1 = deepcopy(parent1)
                child2 = deepcopy(parent2)
            
            # Mutation
            child1.mutate(mutation_rate=0.2, mutation_strength=3)
            child2.mutate(mutation_rate=0.2, mutation_strength=3)
            
            # Reset objectives (need recalculation)
            child1.objectives = None
            child2.objectives = None
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def select_next_generation(self, population):
        """Select next generation from combined population"""
        fronts = self.non_dominated_sort(population)
        
        next_gen = []
        front_idx = 0
        
        while len(next_gen) + len(fronts[front_idx]) <= self.population_size:
            for ind in fronts[front_idx]:
                next_gen.append(ind)
            front_idx += 1
            if front_idx >= len(fronts):
                break
        
        # If we need more individuals from the last front
        if len(next_gen) < self.population_size and front_idx < len(fronts):
            self.calculate_crowding_distance(fronts[front_idx])
            fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
            
            remaining = self.population_size - len(next_gen)
            next_gen.extend(fronts[front_idx][:remaining])
        
        return next_gen
    
    def run(self, num_generations=200):
        """Run NSGA-II optimization"""
        print(f"\nRunning NSGA-II for {num_generations} generations...")
        
        # Evaluate initial population
        self.evaluate_population(self.population)
        fronts = self.non_dominated_sort(self.population)
        
        for gen in range(num_generations):
            # Create offspring
            offspring = self.create_offspring(self.population)
            
            # Evaluate offspring
            self.evaluate_population(offspring)
            
            # Combine populations
            combined = self.population + offspring
            
            # Select next generation
            self.population = self.select_next_generation(combined)
            
            # Update fronts
            fronts = self.non_dominated_sort(self.population)
            for front in fronts:
                self.calculate_crowding_distance(front)
            
            # Record history
            awt_values = [ind.objectives[0] for ind in self.population]
            self.history['generations'].append(gen)
            self.history['best_awt'].append(min(awt_values))
            self.history['avg_awt'].append(np.mean(awt_values))
            self.history['pareto_size'].append(len(fronts[0]))
            
            if (gen + 1) % 50 == 0:
                print(f"  Generation {gen + 1}: Best AWT = {min(awt_values):.2f}s, "
                      f"Pareto front size = {len(fronts[0])}")
        
        print("Optimization complete!")
        
        return fronts[0]  # Return Pareto front
    
    def get_pareto_front(self):
        """Get current Pareto front"""
        fronts = self.non_dominated_sort(self.population)
        return fronts[0] if fronts else []


def visualize_results(optimizer, pareto_front):
    """Create visualizations"""
    print("\nGenerating visualizations...")
    
    # 1. Pareto front (3D scatter)
    fig = plt.figure(figsize=(14, 5))
    
    # 3D Pareto front
    ax1 = fig.add_subplot(131, projection='3d')
    
    awt = [ind.objectives[0] for ind in pareto_front]
    lwr = [ind.objectives[1] for ind in pareto_front]
    energy = [ind.objectives[2] for ind in pareto_front]
    
    scatter = ax1.scatter(awt, lwr, energy, c=awt, cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel('AWT (s)', fontsize=10)
    ax1.set_ylabel('Long Wait Ratio (%)', fontsize=10)
    ax1.set_zlabel('Energy', fontsize=10)
    ax1.set_title('Pareto Front (3D)', fontsize=12)
    plt.colorbar(scatter, ax=ax1, shrink=0.5, label='AWT')
    
    # 2D projection: AWT vs Long Wait Ratio
    ax2 = fig.add_subplot(132)
    ax2.scatter(awt, lwr, c=energy, cmap='plasma', s=60, alpha=0.7, edgecolors='black', linewidths=0.5)
    ax2.set_xlabel('Average Wait Time (s)', fontsize=11)
    ax2.set_ylabel('Long Wait Ratio (%)', fontsize=11)
    ax2.set_title('AWT vs Long Wait Ratio', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 2D projection: AWT vs Energy
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(awt, energy, c=lwr, cmap='coolwarm', s=60, alpha=0.7, 
                           edgecolors='black', linewidths=0.5)
    ax3.set_xlabel('Average Wait Time (s)', fontsize=11)
    ax3.set_ylabel('Energy Consumption', fontsize=11)
    ax3.set_title('AWT vs Energy', fontsize=12)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Long Wait Ratio (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pareto_front.png'), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/pareto_front.png")
    
    # 2. Convergence plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Best AWT over generations
    ax1 = axes[0]
    ax1.plot(optimizer.history['generations'], optimizer.history['best_awt'], 
             'b-', linewidth=1.5, label='Best AWT')
    ax1.plot(optimizer.history['generations'], optimizer.history['avg_awt'], 
             'r--', linewidth=1, alpha=0.7, label='Avg AWT')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Average Wait Time (s)')
    ax1.set_title('Convergence: AWT')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pareto front size
    ax2 = axes[1]
    ax2.plot(optimizer.history['generations'], optimizer.history['pareto_size'], 
             'g-', linewidth=1.5)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Pareto Front Size')
    ax2.set_title('Pareto Front Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Final population distribution
    ax3 = axes[2]
    all_awt = [ind.objectives[0] for ind in optimizer.population]
    ax3.hist(all_awt, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Average Wait Time (s)')
    ax3.set_ylabel('Count')
    ax3.set_title('Final Population AWT Distribution')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'convergence.png'), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/convergence.png")
    
    # 3. Best solutions visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Find best solutions for each objective
    best_awt_idx = np.argmin(awt)
    best_lwr_idx = np.argmin(lwr)
    best_energy_idx = np.argmin(energy)
    
    best_solutions = [
        (pareto_front[best_awt_idx], 'Best AWT'),
        (pareto_front[best_lwr_idx], 'Best Long Wait Ratio'),
        (pareto_front[best_energy_idx], 'Best Energy')
    ]
    
    for ax, (solution, title) in zip(axes, best_solutions):
        floors = solution.get_parking_floors()
        unique_floors, counts = np.unique(floors, return_counts=True)
        
        ax.bar(unique_floors, counts, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Floor')
        ax.set_ylabel('Number of Elevators')
        ax.set_title(f'{title}\nAWT={solution.objectives[0]:.1f}s, LWR={solution.objectives[1]:.1f}%')
        ax.set_xticks(range(optimizer.min_floor, optimizer.max_floor + 1))
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'best_solutions.png'), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/best_solutions.png")


def save_results(optimizer, pareto_front):
    """Save results to files"""
    print("\nSaving results...")
    
    # Save Pareto front solutions
    pareto_data = []
    for i, ind in enumerate(pareto_front):
        pareto_data.append({
            'solution_id': i,
            'parking_floors': ind.get_parking_floors(),
            'AWT': ind.objectives[0],
            'long_wait_ratio': ind.objectives[1],
            'energy': ind.objectives[2]
        })
    
    with open(os.path.join(OUTPUT_DIR, 'pareto_solutions.json'), 'w') as f:
        json.dump(pareto_data, f, indent=2)
    
    # Save as CSV
    df = pd.DataFrame([{
        'solution_id': d['solution_id'],
        'AWT': d['AWT'],
        'long_wait_ratio': d['long_wait_ratio'],
        'energy': d['energy'],
        'parking_floors': str(d['parking_floors'])
    } for d in pareto_data])
    df.to_csv(os.path.join(OUTPUT_DIR, 'pareto_solutions.csv'), index=False)
    
    # Save optimization history
    history_df = pd.DataFrame(optimizer.history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, 'optimization_history.csv'), index=False)
    
    # Save best solutions summary
    best_solutions = {
        'best_awt': None,
        'best_long_wait_ratio': None,
        'best_energy': None,
        'balanced': None
    }
    
    # Find best for each objective
    awt_values = [ind.objectives[0] for ind in pareto_front]
    lwr_values = [ind.objectives[1] for ind in pareto_front]
    energy_values = [ind.objectives[2] for ind in pareto_front]
    
    best_awt_idx = np.argmin(awt_values)
    best_lwr_idx = np.argmin(lwr_values)
    best_energy_idx = np.argmin(energy_values)
    
    best_solutions['best_awt'] = {
        'parking_floors': pareto_front[best_awt_idx].get_parking_floors(),
        'objectives': pareto_front[best_awt_idx].objectives
    }
    best_solutions['best_long_wait_ratio'] = {
        'parking_floors': pareto_front[best_lwr_idx].get_parking_floors(),
        'objectives': pareto_front[best_lwr_idx].objectives
    }
    best_solutions['best_energy'] = {
        'parking_floors': pareto_front[best_energy_idx].get_parking_floors(),
        'objectives': pareto_front[best_energy_idx].objectives
    }
    
    # Find balanced solution (minimize sum of normalized objectives)
    normalized_awt = (np.array(awt_values) - min(awt_values)) / (max(awt_values) - min(awt_values) + 1e-10)
    normalized_lwr = (np.array(lwr_values) - min(lwr_values)) / (max(lwr_values) - min(lwr_values) + 1e-10)
    normalized_energy = (np.array(energy_values) - min(energy_values)) / (max(energy_values) - min(energy_values) + 1e-10)
    
    combined_score = normalized_awt + normalized_lwr + normalized_energy
    balanced_idx = np.argmin(combined_score)
    
    best_solutions['balanced'] = {
        'parking_floors': pareto_front[balanced_idx].get_parking_floors(),
        'objectives': pareto_front[balanced_idx].objectives
    }
    
    with open(os.path.join(OUTPUT_DIR, 'best_solutions.json'), 'w') as f:
        json.dump(best_solutions, f, indent=2)
    
    print(f"  Results saved to {OUTPUT_DIR}/")


def main():
    print("=" * 60)
    print("Task 3: Model B - Multi-Objective Optimization (NSGA-II)")
    print("=" * 60)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    
    params_path = os.path.join(PREPROCESS_DIR, "system_params.json")
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
    else:
        print("Warning: system_params.json not found, using defaults")
        params = {
            "num_elevators": 8,
            "min_floor": 1,
            "max_floor": 14,
            "num_floors": 14,
            "avg_travel_time_per_floor": 3.0,
            "door_open_time": 4.0
        }
    
    demand_path = os.path.join(PREPROCESS_DIR, "floor_call_distribution.csv")
    if os.path.exists(demand_path):
        demand_data = pd.read_csv(demand_path)
    else:
        print("Warning: floor_call_distribution.csv not found")
        demand_data = None
    
    print(f"System parameters:")
    print(f"  - Elevators: {params['num_elevators']}")
    print(f"  - Floors: {params['min_floor']}-{params['max_floor']}")
    
    # Initialize objective calculator
    calculator = ObjectiveCalculator(params, demand_data)
    
    # Initialize and run NSGA-II
    optimizer = NSGA2Optimizer(params, calculator, population_size=100)
    pareto_front = optimizer.run(num_generations=200)
    
    # Print results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    
    print(f"\nPareto Front Size: {len(pareto_front)} solutions")
    
    awt_values = [ind.objectives[0] for ind in pareto_front]
    lwr_values = [ind.objectives[1] for ind in pareto_front]
    energy_values = [ind.objectives[2] for ind in pareto_front]
    
    print(f"\nObjective Ranges on Pareto Front:")
    print(f"  AWT: {min(awt_values):.2f}s - {max(awt_values):.2f}s")
    print(f"  Long Wait Ratio: {min(lwr_values):.2f}% - {max(lwr_values):.2f}%")
    print(f"  Energy: {min(energy_values):.2f} - {max(energy_values):.2f}")
    
    # Best solutions
    best_awt_idx = np.argmin(awt_values)
    best_lwr_idx = np.argmin(lwr_values)
    best_energy_idx = np.argmin(energy_values)
    
    print(f"\nBest Solutions:")
    print(f"  Best AWT: {pareto_front[best_awt_idx].objectives}")
    print(f"    Floors: {pareto_front[best_awt_idx].get_parking_floors()}")
    print(f"  Best LWR: {pareto_front[best_lwr_idx].objectives}")
    print(f"    Floors: {pareto_front[best_lwr_idx].get_parking_floors()}")
    print(f"  Best Energy: {pareto_front[best_energy_idx].objectives}")
    print(f"    Floors: {pareto_front[best_energy_idx].get_parking_floors()}")
    
    # Visualize
    visualize_results(optimizer, pareto_front)
    
    # Save results
    save_results(optimizer, pareto_front)
    
    print("\n" + "=" * 60)
    print("Model B (NSGA-II) complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
