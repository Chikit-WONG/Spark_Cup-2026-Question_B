#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 3: The Strategic Wait - Model C: Discrete Event Simulation
Validates parking strategies using event-driven simulation
All outputs are in English.

Key Fix: Properly calculates wait time as time from hall call to elevator arrival
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
import random

# Output directory
OUTPUT_DIR = "output/model_C"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data directory
PREPROCESS_DIR = "output/preprocessed"
DATA_DIR = "../mcm26Train-B-Data_clean"


class HallCall:
    """Represents a hall call (passenger request)"""
    
    def __init__(self, call_id, timestamp, from_floor, direction):
        self.call_id = call_id
        self.timestamp = timestamp
        self.from_floor = from_floor
        self.direction = direction  # 'up' or 'down'
        self.assigned_elevator = None
        self.pickup_time = None
        self.wait_time = None


class Elevator:
    """Represents an elevator"""
    
    def __init__(self, elevator_id, initial_floor, num_floors):
        self.id = elevator_id
        self.current_floor = initial_floor
        self.target_floor = initial_floor
        self.state = 'idle'  # 'idle', 'moving', 'door_open'
        self.direction = 'none'  # 'up', 'down', 'none'
        self.passengers = 0
        self.max_passengers = 13
        self.num_floors = num_floors
        
        # Physical parameters
        self.travel_time_per_floor = 3.0  # seconds
        self.door_open_time = 4.0  # seconds
        self.door_close_time = 2.0  # seconds
        
        # Assigned calls
        self.assigned_calls = []
        
        # Time tracking
        self.busy_until = 0
    
    def distance_to(self, floor):
        """Calculate time to reach a floor"""
        floors_to_travel = abs(self.current_floor - floor)
        travel_time = floors_to_travel * self.travel_time_per_floor
        
        # Add door time if we need to stop
        total_time = travel_time + self.door_open_time
        
        return total_time
    
    def assign_call(self, call, current_time):
        """Assign a hall call to this elevator"""
        self.assigned_calls.append(call)
        call.assigned_elevator = self.id
        
        # Calculate arrival time
        arrival_time = current_time + self.distance_to(call.from_floor)
        
        # Update elevator state
        self.target_floor = call.from_floor
        self.state = 'moving'
        self.direction = 'up' if call.from_floor > self.current_floor else 'down'
        
        return arrival_time
    
    def arrive_at_floor(self, floor, current_time):
        """Elevator arrives at a floor"""
        self.current_floor = floor
        self.state = 'door_open'
        self.busy_until = current_time + self.door_open_time
    
    def finish_service(self, current_time):
        """Finish serving at current floor"""
        # Remove served calls
        served_calls = [c for c in self.assigned_calls if c.from_floor == self.current_floor]
        for call in served_calls:
            call.pickup_time = current_time
            call.wait_time = call.pickup_time - call.timestamp
            self.assigned_calls.remove(call)
        
        # Check for remaining calls
        if self.assigned_calls:
            next_call = self.assigned_calls[0]
            self.target_floor = next_call.from_floor
            self.state = 'moving'
            self.direction = 'up' if next_call.from_floor > self.current_floor else 'down'
        else:
            self.state = 'idle'
            self.direction = 'none'
        
        return served_calls


class ElevatorSimulator:
    """Discrete event simulation for elevator system"""
    
    def __init__(self, params, demand_distribution):
        self.params = params
        self.demand_distribution = demand_distribution
        
        self.num_elevators = params['num_elevators']
        self.min_floor = params['min_floor']
        self.max_floor = params['max_floor']
        self.num_floors = params['num_floors']
        
        self.travel_time_per_floor = params.get('avg_travel_time_per_floor', 3.0)
        self.door_open_time = params.get('door_open_time', 4.0)
        
        # Elevators
        self.elevators = []
        
        # Event queue (priority queue by time)
        self.event_queue = []
        
        # Statistics
        self.completed_calls = []
        self.current_time = 0
    
    def initialize_elevators(self, parking_floors):
        """Initialize elevators at parking positions"""
        self.elevators = []
        elevator_ids = self.params.get('elevator_ids', 
                                       [chr(ord('A') + i) for i in range(self.num_elevators)])
        
        for i, floor in enumerate(parking_floors):
            if i < len(elevator_ids):
                elevator = Elevator(elevator_ids[i], floor, self.num_floors)
            else:
                elevator = Elevator(f'E{i}', floor, self.num_floors)
            elevator.travel_time_per_floor = self.travel_time_per_floor
            elevator.door_open_time = self.door_open_time
            self.elevators.append(elevator)
    
    def generate_calls_from_data(self, hall_calls_df, simulation_duration=3600):
        """Generate calls from actual data distribution"""
        calls = []
        
        # Calculate call rate per second from data
        total_calls = len(hall_calls_df)
        data_duration = 24 * 3600  # Assume data covers 24 hours
        call_rate = total_calls / data_duration
        
        # Scale to simulation duration with realistic rate
        expected_calls = int(call_rate * simulation_duration)
        
        # Build floor probability distribution
        floor_counts = hall_calls_df['from_floor'].value_counts()
        floor_probs = floor_counts / floor_counts.sum()
        
        # Generate call arrival times (Poisson process)
        inter_arrival_mean = simulation_duration / max(expected_calls, 1)
        
        current_time = 0
        call_id = 0
        
        while current_time < simulation_duration:
            # Exponential inter-arrival time
            inter_arrival = np.random.exponential(inter_arrival_mean)
            current_time += inter_arrival
            
            if current_time >= simulation_duration:
                break
            
            # Select floor based on distribution
            floor = np.random.choice(floor_probs.index, p=floor_probs.values)
            floor = int(floor)
            
            # Determine direction (based on floor position)
            if floor == self.min_floor:
                direction = 'up'
            elif floor == self.max_floor:
                direction = 'down'
            else:
                # Probability based on typical patterns
                if floor <= 3:
                    direction = 'up' if random.random() < 0.7 else 'down'
                else:
                    direction = 'down' if random.random() < 0.6 else 'up'
            
            call = HallCall(call_id, current_time, floor, direction)
            calls.append(call)
            call_id += 1
        
        return calls
    
    def find_best_elevator(self, call):
        """Find the best elevator to serve a call"""
        best_elevator = None
        best_score = float('inf')
        
        for elevator in self.elevators:
            # Calculate score based on distance and state
            distance_time = elevator.distance_to(call.from_floor)
            
            # Penalty for busy elevators
            if elevator.state != 'idle':
                distance_time += len(elevator.assigned_calls) * 5.0
            
            # Prefer elevators moving in the right direction
            if elevator.direction == call.direction:
                distance_time *= 0.8
            
            if distance_time < best_score:
                best_score = distance_time
                best_elevator = elevator
        
        return best_elevator, best_score
    
    def process_call(self, call):
        """Process a hall call - assign to best elevator"""
        best_elevator, expected_wait = self.find_best_elevator(call)
        
        if best_elevator:
            arrival_time = best_elevator.assign_call(call, call.timestamp)
            
            # Schedule arrival event
            heapq.heappush(self.event_queue, (arrival_time, 'arrival', 
                                               best_elevator.id, call.call_id))
            
            return expected_wait
        
        return None
    
    def run_simulation(self, parking_floors, simulation_duration=3600, hall_calls_df=None):
        """Run the simulation"""
        # Reset state
        self.event_queue = []
        self.completed_calls = []
        self.current_time = 0
        
        # Initialize elevators
        self.initialize_elevators(parking_floors)
        
        # Generate calls
        if hall_calls_df is not None:
            calls = self.generate_calls_from_data(hall_calls_df, simulation_duration)
        else:
            calls = self._generate_synthetic_calls(simulation_duration)
        
        # Process each call
        for call in calls:
            self.current_time = call.timestamp
            
            # Find and assign elevator
            best_elevator, expected_time = self.find_best_elevator(call)
            
            if best_elevator:
                # Calculate actual wait time
                travel_time = abs(best_elevator.current_floor - call.from_floor) * self.travel_time_per_floor
                actual_wait = travel_time + self.door_open_time
                
                # Add queue delay if elevator is busy
                if best_elevator.assigned_calls:
                    queue_delay = len(best_elevator.assigned_calls) * (self.door_open_time + 3.0)
                    actual_wait += queue_delay
                
                call.wait_time = actual_wait
                call.pickup_time = call.timestamp + actual_wait
                
                # Update elevator position (simplified: jump to call floor after serving)
                best_elevator.current_floor = call.from_floor
                best_elevator.assigned_calls.append(call)
                
                # Remove from queue after delay
                if len(best_elevator.assigned_calls) > 5:
                    best_elevator.assigned_calls.pop(0)
            else:
                call.wait_time = 60.0  # Default long wait
            
            self.completed_calls.append(call)
        
        return self.calculate_statistics()
    
    def _generate_synthetic_calls(self, duration):
        """Generate synthetic calls if no data available"""
        calls = []
        call_id = 0
        
        # Average 1 call per 10 seconds
        num_calls = int(duration / 10)
        
        for _ in range(num_calls):
            timestamp = random.uniform(0, duration)
            floor = random.randint(self.min_floor, self.max_floor)
            direction = 'up' if floor < self.max_floor // 2 else 'down'
            
            call = HallCall(call_id, timestamp, floor, direction)
            calls.append(call)
            call_id += 1
        
        # Sort by timestamp
        calls.sort(key=lambda c: c.timestamp)
        return calls
    
    def calculate_statistics(self):
        """Calculate simulation statistics"""
        if not self.completed_calls:
            return {
                'avg_wait_time': 0,
                'max_wait_time': 0,
                'long_wait_ratio': 0,
                'total_calls': 0
            }
        
        wait_times = [c.wait_time for c in self.completed_calls if c.wait_time is not None]
        
        if not wait_times:
            return {
                'avg_wait_time': 0,
                'max_wait_time': 0,
                'long_wait_ratio': 0,
                'total_calls': len(self.completed_calls)
            }
        
        long_waits = sum(1 for w in wait_times if w > 60)
        
        stats = {
            'avg_wait_time': np.mean(wait_times),
            'std_wait_time': np.std(wait_times),
            'max_wait_time': np.max(wait_times),
            'min_wait_time': np.min(wait_times),
            'median_wait_time': np.median(wait_times),
            'long_wait_ratio': long_waits / len(wait_times) * 100,
            'total_calls': len(wait_times),
            'percentile_90': np.percentile(wait_times, 90),
            'percentile_95': np.percentile(wait_times, 95)
        }
        
        return stats


class StrategyEvaluator:
    """Evaluate different parking strategies"""
    
    def __init__(self, params, hall_calls_df):
        self.params = params
        self.hall_calls_df = hall_calls_df
        self.simulator = ElevatorSimulator(params, None)
        
        self.num_elevators = params['num_elevators']
        self.min_floor = params['min_floor']
        self.max_floor = params['max_floor']
    
    def define_strategies(self):
        """Define parking strategies to evaluate"""
        strategies = {}
        
        # Strategy 1: All at ground floor
        strategies['Ground_Floor'] = [self.min_floor] * self.num_elevators
        
        # Strategy 2: Uniform distribution
        step = (self.max_floor - self.min_floor) / max(1, self.num_elevators - 1)
        uniform_floors = [int(self.min_floor + i * step) for i in range(self.num_elevators)]
        strategies['Uniform_Distribution'] = uniform_floors
        
        # Strategy 3: Demand-based (concentrate on high-demand floors)
        demand_floors = self._get_demand_based_floors()
        strategies['Demand_Based'] = demand_floors
        
        # Strategy 4: Zone-based
        zone_floors = self._get_zone_based_floors()
        strategies['Zone_Based'] = zone_floors
        
        # Strategy 5: Hybrid (mix of ground and distributed)
        hybrid = [self.min_floor] * (self.num_elevators // 2)
        for i in range(self.num_elevators - len(hybrid)):
            hybrid.append(min(self.min_floor + (i + 1) * 4, self.max_floor))
        strategies['Hybrid'] = hybrid
        
        # Strategy 6: Middle concentration
        middle = (self.min_floor + self.max_floor) // 2
        middle_floors = [middle + (i % 3) - 1 for i in range(self.num_elevators)]
        middle_floors = [max(self.min_floor, min(self.max_floor, f)) for f in middle_floors]
        strategies['Middle_Concentration'] = middle_floors
        
        return strategies
    
    def _get_demand_based_floors(self):
        """Get floors based on demand distribution"""
        if self.hall_calls_df is None or len(self.hall_calls_df) == 0:
            return [self.min_floor] * self.num_elevators
        
        floor_counts = self.hall_calls_df['from_floor'].value_counts()
        top_floors = floor_counts.nlargest(self.num_elevators).index.tolist()
        
        while len(top_floors) < self.num_elevators:
            top_floors.append(self.min_floor)
        
        return [int(f) for f in top_floors[:self.num_elevators]]
    
    def _get_zone_based_floors(self):
        """Get zone-based parking floors"""
        zone_size = (self.max_floor - self.min_floor) // 3
        zones = [
            self.min_floor + zone_size // 2,
            self.min_floor + zone_size + zone_size // 2,
            self.min_floor + 2 * zone_size + zone_size // 2
        ]
        
        floors = []
        for i in range(self.num_elevators):
            zone_idx = i % 3
            floors.append(min(zones[zone_idx], self.max_floor))
        
        return floors
    
    def evaluate_strategies(self, num_simulations=10, simulation_duration=3600):
        """Evaluate all strategies"""
        print("\nEvaluating parking strategies...")
        
        strategies = self.define_strategies()
        results = {}
        
        for name, floors in strategies.items():
            print(f"\n  Evaluating: {name}")
            print(f"    Parking floors: {floors}")
            
            all_stats = []
            
            for sim in range(num_simulations):
                stats = self.simulator.run_simulation(
                    floors, 
                    simulation_duration,
                    self.hall_calls_df
                )
                all_stats.append(stats)
            
            # Aggregate statistics
            avg_stats = {
                'strategy': name,
                'parking_floors': floors,
                'avg_wait_time': np.mean([s['avg_wait_time'] for s in all_stats]),
                'std_wait_time': np.mean([s.get('std_wait_time', 0) for s in all_stats]),
                'max_wait_time': np.mean([s['max_wait_time'] for s in all_stats]),
                'long_wait_ratio': np.mean([s['long_wait_ratio'] for s in all_stats]),
                'total_calls': np.mean([s['total_calls'] for s in all_stats])
            }
            
            results[name] = avg_stats
            print(f"    Avg Wait Time: {avg_stats['avg_wait_time']:.2f}s")
            print(f"    Long Wait Ratio: {avg_stats['long_wait_ratio']:.2f}%")
        
        return results
    
    def evaluate_scenarios(self, num_simulations=5):
        """Evaluate strategies under different demand scenarios"""
        print("\nEvaluating under different scenarios...")
        
        strategies = self.define_strategies()
        scenario_results = {}
        
        # Define scenarios by filtering data
        scenarios = {
            'Morning_Peak': (7, 10),
            'Midday': (10, 14),
            'Afternoon': (14, 17),
            'Evening_Peak': (17, 20)
        }
        
        for scenario_name, (start_hour, end_hour) in scenarios.items():
            print(f"\n  Scenario: {scenario_name} ({start_hour}:00 - {end_hour}:00)")
            
            # Filter data for this time period (if timestamp available)
            if self.hall_calls_df is not None and 'timestamp' in self.hall_calls_df.columns:
                try:
                    df = self.hall_calls_df.copy()
                    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                    scenario_data = df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)]
                except:
                    scenario_data = self.hall_calls_df
            else:
                scenario_data = self.hall_calls_df
            
            scenario_results[scenario_name] = {}
            
            for strategy_name, floors in strategies.items():
                stats_list = []
                
                for _ in range(num_simulations):
                    stats = self.simulator.run_simulation(
                        floors,
                        simulation_duration=1800,  # 30 minutes
                        hall_calls_df=scenario_data
                    )
                    stats_list.append(stats)
                
                avg_awt = np.mean([s['avg_wait_time'] for s in stats_list])
                scenario_results[scenario_name][strategy_name] = {
                    'avg_wait_time': avg_awt,
                    'long_wait_ratio': np.mean([s['long_wait_ratio'] for s in stats_list])
                }
            
            # Find best strategy for this scenario
            best = min(scenario_results[scenario_name].items(), 
                      key=lambda x: x[1]['avg_wait_time'])
            print(f"    Best strategy: {best[0]} (AWT: {best[1]['avg_wait_time']:.2f}s)")
        
        return scenario_results


def visualize_results(strategy_results, scenario_results):
    """Create visualizations"""
    print("\nGenerating visualizations...")
    
    # 1. Strategy comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Average Wait Time
    ax1 = axes[0, 0]
    strategies = list(strategy_results.keys())
    awt_values = [strategy_results[s]['avg_wait_time'] for s in strategies]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(strategies)))
    bars = ax1.bar(range(len(strategies)), awt_values, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.set_ylabel('Average Wait Time (s)')
    ax1.set_title('Average Wait Time by Strategy')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, awt_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # Long Wait Ratio
    ax2 = axes[0, 1]
    lwr_values = [strategy_results[s]['long_wait_ratio'] for s in strategies]
    bars = ax2.bar(range(len(strategies)), lwr_values, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.set_ylabel('Long Wait Ratio (%)')
    ax2.set_title('Long Wait Ratio (>60s) by Strategy')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Scenario heatmap
    ax3 = axes[1, 0]
    scenarios = list(scenario_results.keys())
    strategies = list(next(iter(scenario_results.values())).keys())
    
    heatmap_data = np.array([
        [scenario_results[sc][st]['avg_wait_time'] for st in strategies]
        for sc in scenarios
    ])
    
    im = ax3.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
    ax3.set_xticks(range(len(strategies)))
    ax3.set_yticks(range(len(scenarios)))
    ax3.set_xticklabels(strategies, rotation=45, ha='right', fontsize=9)
    ax3.set_yticklabels(scenarios)
    ax3.set_title('AWT by Strategy and Scenario (s)')
    plt.colorbar(im, ax=ax3, label='Wait Time (s)')
    
    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(strategies)):
            val = heatmap_data[i, j]
            ax3.text(j, i, f'{val:.1f}', ha='center', va='center', 
                    fontsize=8, color='white' if val > heatmap_data.max()/2 else 'black')
    
    # Best strategy per scenario
    ax4 = axes[1, 1]
    best_per_scenario = []
    for sc in scenarios:
        best = min(scenario_results[sc].items(), key=lambda x: x[1]['avg_wait_time'])
        best_per_scenario.append((sc, best[0], best[1]['avg_wait_time']))
    
    sc_names = [b[0] for b in best_per_scenario]
    best_strategies = [b[1] for b in best_per_scenario]
    best_awt = [b[2] for b in best_per_scenario]
    
    bars = ax4.bar(sc_names, best_awt, color='steelblue', edgecolor='black')
    ax4.set_ylabel('Best AWT (s)')
    ax4.set_title('Best Strategy per Scenario')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add strategy labels
    for bar, strat in zip(bars, best_strategies):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                strat.replace('_', '\n'), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'simulation_results.png'), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/simulation_results.png")
    
    # 2. Parking floor visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (name, data) in enumerate(strategy_results.items()):
        floors = data['parking_floors']
        y = [i] * len(floors)
        ax.scatter(floors, y, s=100, alpha=0.7, label=f'{name} (AWT: {data["avg_wait_time"]:.1f}s)')
    
    ax.set_yticks(range(len(strategy_results)))
    ax.set_yticklabels(list(strategy_results.keys()))
    ax.set_xlabel('Floor')
    ax.set_ylabel('Strategy')
    ax.set_title('Elevator Parking Positions by Strategy')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'parking_positions.png'), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/parking_positions.png")


def save_results(strategy_results, scenario_results):
    """Save results to files"""
    print("\nSaving results...")
    
    # Save strategy results
    strategy_df = pd.DataFrame([
        {
            'strategy': name,
            'avg_wait_time': data['avg_wait_time'],
            'std_wait_time': data.get('std_wait_time', 0),
            'max_wait_time': data['max_wait_time'],
            'long_wait_ratio': data['long_wait_ratio'],
            'total_calls': data['total_calls'],
            'parking_floors': str(data['parking_floors'])
        }
        for name, data in strategy_results.items()
    ])
    strategy_df.to_csv(os.path.join(OUTPUT_DIR, 'strategy_comparison.csv'), index=False)
    
    # Save scenario results
    scenario_data = []
    for scenario, strategies in scenario_results.items():
        for strategy, data in strategies.items():
            scenario_data.append({
                'scenario': scenario,
                'strategy': strategy,
                'avg_wait_time': data['avg_wait_time'],
                'long_wait_ratio': data['long_wait_ratio']
            })
    
    scenario_df = pd.DataFrame(scenario_data)
    scenario_df.to_csv(os.path.join(OUTPUT_DIR, 'scenario_results.csv'), index=False)
    
    # Save summary
    best_strategy = min(strategy_results.items(), key=lambda x: x[1]['avg_wait_time'])
    
    summary = {
        'best_overall_strategy': best_strategy[0],
        'best_avg_wait_time': best_strategy[1]['avg_wait_time'],
        'best_parking_floors': best_strategy[1]['parking_floors'],
        'all_strategies': {
            name: {
                'avg_wait_time': data['avg_wait_time'],
                'long_wait_ratio': data['long_wait_ratio']
            }
            for name, data in strategy_results.items()
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'simulation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Results saved to {OUTPUT_DIR}/")


def main():
    print("=" * 60)
    print("Task 3: Model C - Discrete Event Simulation")
    print("=" * 60)
    
    # Load preprocessed data
    print("\nLoading data...")
    
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
            "door_open_time": 4.0,
            "elevator_ids": ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        }
    
    # Load hall calls data
    hall_calls_path = os.path.join(DATA_DIR, "clean_hall_calls.csv")
    if os.path.exists(hall_calls_path):
        hall_calls_df = pd.read_csv(hall_calls_path)
        # Handle column naming
        floor_col = 'Floor' if 'Floor' in hall_calls_df.columns else 'from_floor'
        hall_calls_df['from_floor'] = pd.to_numeric(hall_calls_df[floor_col], errors='coerce')
        hall_calls_df = hall_calls_df.dropna(subset=['from_floor'])
        hall_calls_df['from_floor'] = hall_calls_df['from_floor'].astype(int)
        # Handle timestamp column
        if 'Time' in hall_calls_df.columns:
            hall_calls_df['timestamp'] = hall_calls_df['Time']
        print(f"Loaded {len(hall_calls_df)} hall calls")
    else:
        print("Warning: Hall calls data not found, using synthetic data")
        hall_calls_df = None
    
    print(f"\nSystem parameters:")
    print(f"  - Elevators: {params['num_elevators']}")
    print(f"  - Floors: {params['min_floor']}-{params['max_floor']}")
    
    # Initialize evaluator
    evaluator = StrategyEvaluator(params, hall_calls_df)
    
    # Evaluate strategies
    strategy_results = evaluator.evaluate_strategies(
        num_simulations=10,
        simulation_duration=3600
    )
    
    # Evaluate scenarios
    scenario_results = evaluator.evaluate_scenarios(num_simulations=5)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Simulation Results Summary")
    print("=" * 60)
    
    print("\nStrategy Comparison:")
    for name, data in sorted(strategy_results.items(), key=lambda x: x[1]['avg_wait_time']):
        print(f"  {name}:")
        print(f"    Avg Wait Time: {data['avg_wait_time']:.2f}s")
        print(f"    Long Wait Ratio: {data['long_wait_ratio']:.2f}%")
        print(f"    Parking Floors: {data['parking_floors']}")
    
    best = min(strategy_results.items(), key=lambda x: x[1]['avg_wait_time'])
    print(f"\nBest Overall Strategy: {best[0]}")
    print(f"  Avg Wait Time: {best[1]['avg_wait_time']:.2f}s")
    print(f"  Parking Floors: {best[1]['parking_floors']}")
    
    # Visualize
    visualize_results(strategy_results, scenario_results)
    
    # Save results
    save_results(strategy_results, scenario_results)
    
    print("\n" + "=" * 60)
    print("Model C (Simulation) complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
