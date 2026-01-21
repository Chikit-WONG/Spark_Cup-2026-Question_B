#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 3: The Strategic Wait - Model A: MDP + Q-Learning Approach
Dynamic Parking Strategy Optimization using Reinforcement Learning
All outputs are in English.
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Output directory
OUTPUT_DIR = "output/model_A"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data directory
PREPROCESS_DIR = "output/preprocessed"

class ElevatorParkingMDP:
    """
    Markov Decision Process for elevator parking optimization
    
    State Space: (time_bucket, demand_level, current_parking_config)
    Action Space: Different parking configurations
    Reward: Negative of estimated waiting time
    """
    
    def __init__(self, params, demand_data):
        self.params = params
        self.demand_data = demand_data
        
        # System parameters
        self.num_elevators = params['num_elevators']
        self.min_floor = params['min_floor']
        self.max_floor = params['max_floor']
        self.num_floors = params['num_floors']
        
        # Time buckets (24 hours -> 4 periods)
        self.time_buckets = ['morning_peak', 'midday', 'evening_peak', 'night']
        self.time_bucket_hours = {
            'morning_peak': list(range(7, 10)),
            'midday': list(range(10, 17)),
            'evening_peak': list(range(17, 20)),
            'night': list(range(20, 24)) + list(range(0, 7))
        }
        
        # Demand levels
        self.demand_levels = ['low', 'medium', 'high']
        
        # Define parking zones
        self.zones = self._define_zones()
        
        # Generate action space (parking configurations)
        self.actions = self._generate_actions()
        
        # State space
        self.states = self._generate_states()
        
        print(f"MDP initialized:")
        print(f"  - Number of states: {len(self.states)}")
        print(f"  - Number of actions: {len(self.actions)}")
        print(f"  - Number of elevators: {self.num_elevators}")
        print(f"  - Floor range: {self.min_floor}-{self.max_floor}")
    
    def _define_zones(self):
        """Define floor zones based on building structure"""
        total_floors = self.max_floor - self.min_floor + 1
        zone_size = total_floors // 3
        
        zones = {
            'low': list(range(self.min_floor, self.min_floor + zone_size)),
            'mid': list(range(self.min_floor + zone_size, self.min_floor + 2 * zone_size)),
            'high': list(range(self.min_floor + 2 * zone_size, self.max_floor + 1))
        }
        
        # Ensure all floors are covered
        zones['high'] = list(range(self.min_floor + 2 * zone_size, self.max_floor + 1))
        
        return zones
    
    def _generate_actions(self):
        """Generate possible parking configurations"""
        actions = []
        
        # Strategy 1: Concentrate at ground floor
        actions.append({
            'name': 'ground_concentrate',
            'floors': [self.min_floor] * self.num_elevators
        })
        
        # Strategy 2: Uniform distribution
        step = max(1, self.num_floors // self.num_elevators)
        uniform_floors = []
        for i in range(self.num_elevators):
            floor = min(self.min_floor + i * step, self.max_floor)
            uniform_floors.append(floor)
        actions.append({
            'name': 'uniform_distribution',
            'floors': uniform_floors
        })
        
        # Strategy 3: High demand floor concentration
        actions.append({
            'name': 'demand_based',
            'floors': self._get_high_demand_floors()
        })
        
        # Strategy 4: Zone-based distribution
        zone_floors = []
        for i, zone_name in enumerate(['low', 'mid', 'high']):
            zone = self.zones[zone_name]
            if zone:
                # Assign proportional number of elevators to each zone
                n_elevators_zone = max(1, self.num_elevators // 3)
                for _ in range(n_elevators_zone):
                    zone_floors.append(zone[len(zone) // 2])  # Middle of zone
        # Fill remaining
        while len(zone_floors) < self.num_elevators:
            zone_floors.append(self.min_floor)
        zone_floors = zone_floors[:self.num_elevators]
        actions.append({
            'name': 'zone_based',
            'floors': zone_floors
        })
        
        # Strategy 5: Hybrid (some at ground, some distributed)
        hybrid_floors = [self.min_floor] * (self.num_elevators // 2)
        remaining = self.num_elevators - len(hybrid_floors)
        for i in range(remaining):
            hybrid_floors.append(self.min_floor + (i + 1) * 3)
        hybrid_floors = [min(f, self.max_floor) for f in hybrid_floors]
        actions.append({
            'name': 'hybrid',
            'floors': hybrid_floors
        })
        
        # Strategy 6: Top-heavy (for buildings with high upper floor demand)
        top_floors = []
        for i in range(self.num_elevators):
            floor = max(self.min_floor, self.max_floor - i * 2)
            top_floors.append(floor)
        actions.append({
            'name': 'top_heavy',
            'floors': top_floors
        })
        
        return actions
    
    def _get_high_demand_floors(self):
        """Get floors with highest demand"""
        if self.demand_data is None or len(self.demand_data) == 0:
            return [self.min_floor] * self.num_elevators
        
        # Sort by demand
        sorted_demand = self.demand_data.nlargest(self.num_elevators, 'call_count')
        floors = sorted_demand['floor'].tolist()
        
        # Ensure we have enough floors
        while len(floors) < self.num_elevators:
            floors.append(self.min_floor)
        
        return [int(f) for f in floors[:self.num_elevators]]
    
    def _generate_states(self):
        """Generate state space"""
        states = []
        for time_bucket in self.time_buckets:
            for demand_level in self.demand_levels:
                states.append((time_bucket, demand_level))
        return states
    
    def get_state(self, hour, current_demand):
        """Convert current observation to state"""
        # Determine time bucket
        time_bucket = 'night'
        for bucket, hours in self.time_bucket_hours.items():
            if hour in hours:
                time_bucket = bucket
                break
        
        # Determine demand level
        if current_demand < 0.3:
            demand_level = 'low'
        elif current_demand < 0.7:
            demand_level = 'medium'
        else:
            demand_level = 'high'
        
        return (time_bucket, demand_level)
    
    def calculate_expected_wait_time(self, action, state):
        """
        Calculate expected wait time for a parking configuration
        Based on average distance from parking floors to call floors
        """
        time_bucket, demand_level = state
        parking_floors = action['floors']
        
        # Get demand distribution for this time bucket
        hours = self.time_bucket_hours[time_bucket]
        
        # Weight floors by demand
        floor_weights = {}
        for floor in range(self.min_floor, self.max_floor + 1):
            if self.demand_data is not None and floor in self.demand_data['floor'].values:
                weight = self.demand_data[self.demand_data['floor'] == floor]['call_count'].values[0]
            else:
                weight = 1.0
            floor_weights[floor] = weight
        
        # Normalize weights
        total_weight = sum(floor_weights.values())
        if total_weight > 0:
            floor_weights = {k: v / total_weight for k, v in floor_weights.items()}
        
        # Calculate weighted average distance
        avg_distance = 0
        for floor in range(self.min_floor, self.max_floor + 1):
            # Find nearest elevator
            min_dist = min(abs(floor - pf) for pf in parking_floors)
            avg_distance += floor_weights.get(floor, 0) * min_dist
        
        # Convert distance to time (3 seconds per floor + door time)
        travel_time_per_floor = self.params.get('avg_travel_time_per_floor', 3.0)
        door_time = self.params.get('door_open_time', 4.0)
        
        expected_wait = avg_distance * travel_time_per_floor + door_time
        
        # Apply demand level multiplier
        demand_multiplier = {'low': 0.8, 'medium': 1.0, 'high': 1.3}
        expected_wait *= demand_multiplier[demand_level]
        
        return expected_wait
    
    def get_reward(self, action, state):
        """Calculate reward for taking action in state"""
        expected_wait = self.calculate_expected_wait_time(action, state)
        
        # Negative reward (we want to minimize wait time)
        reward = -expected_wait
        
        # Bonus for balanced distribution
        floors = action['floors']
        floor_std = np.std(floors) if len(floors) > 1 else 0
        balance_bonus = min(2.0, floor_std / 3)  # Small bonus for spread
        
        return reward + balance_bonus


class QLearningAgent:
    """Q-Learning agent for elevator parking optimization"""
    
    def __init__(self, mdp, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        self.mdp = mdp
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Training history
        self.rewards_history = []
        self.epsilon_history = []
    
    def get_action(self, state, explore=True):
        """Select action using epsilon-greedy policy"""
        if explore and random.random() < self.epsilon:
            return random.choice(self.mdp.actions)
        else:
            # Greedy selection
            state_key = str(state)
            best_action = max(self.mdp.actions, 
                            key=lambda a: self.q_table[state_key][a['name']])
            return best_action
    
    def update(self, state, action, reward, next_state):
        """Update Q-value"""
        state_key = str(state)
        next_state_key = str(next_state)
        action_name = action['name']
        
        # Q-learning update
        current_q = self.q_table[state_key][action_name]
        next_max_q = max(self.q_table[next_state_key][a['name']] 
                        for a in self.mdp.actions) if self.mdp.actions else 0
        
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action_name] = new_q
    
    def train(self, num_episodes=5000):
        """Train the agent"""
        print(f"\nTraining Q-Learning agent for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Random initial state
            state = random.choice(self.mdp.states)
            
            episode_reward = 0
            
            # Episode loop (simulate 24 hours)
            for step in range(24):
                # Select action
                action = self.get_action(state, explore=True)
                
                # Get reward
                reward = self.mdp.get_reward(action, state)
                episode_reward += reward
                
                # Transition to next state (simulate time progression)
                next_hour = (step + 1) % 24
                next_demand = random.random()  # Random demand for training
                next_state = self.mdp.get_state(next_hour, next_demand)
                
                # Update Q-table
                self.update(state, action, reward, next_state)
                
                state = next_state
            
            self.rewards_history.append(episode_reward)
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.999)
            self.epsilon_history.append(self.epsilon)
            
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(self.rewards_history[-1000:])
                print(f"  Episode {episode + 1}: avg reward = {avg_reward:.2f}, epsilon = {self.epsilon:.4f}")
        
        print("Training complete!")
    
    def get_policy(self):
        """Extract the learned policy"""
        policy = {}
        for state in self.mdp.states:
            state_key = str(state)
            best_action = max(self.mdp.actions, 
                            key=lambda a: self.q_table[state_key][a['name']])
            policy[state] = best_action
        return policy
    
    def get_optimal_parking(self, hour, demand_level):
        """Get optimal parking configuration for given conditions"""
        # Convert to demand value
        demand_values = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        demand = demand_values.get(demand_level, 0.5)
        
        state = self.mdp.get_state(hour, demand)
        return self.get_action(state, explore=False)


def evaluate_policy(agent, mdp, num_simulations=100):
    """Evaluate the learned policy"""
    print("\nEvaluating learned policy...")
    
    results = {
        'by_time_bucket': {},
        'by_demand_level': {},
        'overall': {}
    }
    
    wait_times = []
    
    for _ in range(num_simulations):
        for state in mdp.states:
            action = agent.get_action(state, explore=False)
            wait = mdp.calculate_expected_wait_time(action, state)
            wait_times.append({
                'state': state,
                'action': action['name'],
                'wait_time': wait
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(wait_times)
    df['time_bucket'] = df['state'].apply(lambda x: x[0])
    df['demand_level'] = df['state'].apply(lambda x: x[1])
    
    # Aggregate by time bucket
    for bucket in mdp.time_buckets:
        bucket_data = df[df['time_bucket'] == bucket]
        results['by_time_bucket'][bucket] = {
            'avg_wait': bucket_data['wait_time'].mean(),
            'std_wait': bucket_data['wait_time'].std(),
            'best_action': bucket_data.groupby('action')['wait_time'].mean().idxmin()
        }
    
    # Aggregate by demand level
    for level in mdp.demand_levels:
        level_data = df[df['demand_level'] == level]
        results['by_demand_level'][level] = {
            'avg_wait': level_data['wait_time'].mean(),
            'std_wait': level_data['wait_time'].std(),
            'best_action': level_data.groupby('action')['wait_time'].mean().idxmin()
        }
    
    # Overall
    results['overall'] = {
        'avg_wait': df['wait_time'].mean(),
        'std_wait': df['wait_time'].std(),
        'min_wait': df['wait_time'].min(),
        'max_wait': df['wait_time'].max()
    }
    
    return results


def visualize_results(agent, mdp, results):
    """Create visualizations"""
    print("\nGenerating visualizations...")
    
    # 1. Training curve
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training rewards
    ax1 = axes[0, 0]
    window = 100
    smoothed_rewards = pd.Series(agent.rewards_history).rolling(window).mean()
    ax1.plot(smoothed_rewards, color='blue', linewidth=1)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Q-Learning Training Progress')
    ax1.grid(True, alpha=0.3)
    
    # Epsilon decay
    ax2 = axes[0, 1]
    ax2.plot(agent.epsilon_history, color='green', linewidth=1)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate Decay')
    ax2.grid(True, alpha=0.3)
    
    # Wait time by time bucket
    ax3 = axes[1, 0]
    buckets = list(results['by_time_bucket'].keys())
    wait_means = [results['by_time_bucket'][b]['avg_wait'] for b in buckets]
    wait_stds = [results['by_time_bucket'][b]['std_wait'] for b in buckets]
    x = np.arange(len(buckets))
    bars = ax3.bar(x, wait_means, yerr=wait_stds, capsize=5, color='steelblue', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(buckets, rotation=45)
    ax3.set_ylabel('Expected Wait Time (s)')
    ax3.set_title('Wait Time by Time Period')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Wait time by demand level
    ax4 = axes[1, 1]
    levels = list(results['by_demand_level'].keys())
    level_means = [results['by_demand_level'][l]['avg_wait'] for l in levels]
    level_stds = [results['by_demand_level'][l]['std_wait'] for l in levels]
    x = np.arange(len(levels))
    colors = ['lightgreen', 'gold', 'salmon']
    ax4.bar(x, level_means, yerr=level_stds, capsize=5, color=colors, alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(levels)
    ax4.set_ylabel('Expected Wait Time (s)')
    ax4.set_title('Wait Time by Demand Level')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'qlearning_results.png'), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/qlearning_results.png")
    
    # 2. Policy heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    policy = agent.get_policy()
    action_names = [a['name'] for a in mdp.actions]
    
    # Create matrix
    matrix = np.zeros((len(mdp.demand_levels), len(mdp.time_buckets)))
    for i, demand in enumerate(mdp.demand_levels):
        for j, time in enumerate(mdp.time_buckets):
            state = (time, demand)
            action = policy[state]
            matrix[i, j] = action_names.index(action['name'])
    
    im = ax.imshow(matrix, cmap='Set3', aspect='auto')
    ax.set_xticks(np.arange(len(mdp.time_buckets)))
    ax.set_yticks(np.arange(len(mdp.demand_levels)))
    ax.set_xticklabels(mdp.time_buckets, rotation=45)
    ax.set_yticklabels(mdp.demand_levels)
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Demand Level')
    ax.set_title('Optimal Parking Strategy (Policy Map)')
    
    # Add text annotations
    for i in range(len(mdp.demand_levels)):
        for j in range(len(mdp.time_buckets)):
            state = (mdp.time_buckets[j], mdp.demand_levels[i])
            action = policy[state]['name']
            ax.text(j, i, action.replace('_', '\n'), ha='center', va='center', 
                   fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'policy_heatmap.png'), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/policy_heatmap.png")
    
    # 3. Parking floor distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, action in enumerate(mdp.actions):
        floors = action['floors']
        y = [i] * len(floors)
        ax.scatter(floors, y, s=100, alpha=0.7, label=action['name'])
    
    ax.set_yticks(range(len(mdp.actions)))
    ax.set_yticklabels([a['name'] for a in mdp.actions])
    ax.set_xlabel('Floor')
    ax.set_ylabel('Strategy')
    ax.set_title('Elevator Parking Floor Distribution by Strategy')
    ax.set_xlim(mdp.min_floor - 0.5, mdp.max_floor + 0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'parking_distributions.png'), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/parking_distributions.png")


def save_results(agent, mdp, results):
    """Save results to files"""
    print("\nSaving results...")
    
    # Save policy
    policy = agent.get_policy()
    policy_dict = {}
    for state, action in policy.items():
        policy_dict[str(state)] = {
            'action_name': action['name'],
            'parking_floors': action['floors']
        }
    
    with open(os.path.join(OUTPUT_DIR, 'learned_policy.json'), 'w') as f:
        json.dump(policy_dict, f, indent=2)
    
    # Save evaluation results
    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save Q-table summary
    q_summary = {}
    for state in mdp.states:
        state_key = str(state)
        q_summary[state_key] = dict(agent.q_table[state_key])
    
    with open(os.path.join(OUTPUT_DIR, 'q_table.json'), 'w') as f:
        json.dump(q_summary, f, indent=2)
    
    print(f"  Results saved to {OUTPUT_DIR}/")


def main():
    print("=" * 60)
    print("Task 3: Model A - MDP + Q-Learning Approach")
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
    
    # Initialize MDP
    mdp = ElevatorParkingMDP(params, demand_data)
    
    # Initialize and train Q-Learning agent
    agent = QLearningAgent(mdp, learning_rate=0.1, discount_factor=0.95, epsilon=0.3)
    agent.train(num_episodes=5000)
    
    # Evaluate policy
    results = evaluate_policy(agent, mdp)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    print(f"\nOverall Performance:")
    print(f"  Average Wait Time: {results['overall']['avg_wait']:.2f} seconds")
    print(f"  Wait Time Std Dev: {results['overall']['std_wait']:.2f} seconds")
    
    print(f"\nBy Time Period:")
    for bucket, data in results['by_time_bucket'].items():
        print(f"  {bucket}: {data['avg_wait']:.2f}s (best strategy: {data['best_action']})")
    
    print(f"\nBy Demand Level:")
    for level, data in results['by_demand_level'].items():
        print(f"  {level}: {data['avg_wait']:.2f}s (best strategy: {data['best_action']})")
    
    print(f"\nLearned Policy:")
    policy = agent.get_policy()
    for state, action in policy.items():
        print(f"  {state} -> {action['name']} (floors: {action['floors']})")
    
    # Visualize
    visualize_results(agent, mdp, results)
    
    # Save results
    save_results(agent, mdp, results)
    
    print("\n" + "=" * 60)
    print("Model A (Q-Learning) complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
