#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 3: The Strategic Wait - Data Preprocessing
This script processes elevator data to extract features for dynamic parking strategy optimization.
All outputs are in English.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# Output directory
OUTPUT_DIR = "output/preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data directory
DATA_DIR = "../mcm26Train-B-Data_clean"

def load_data():
    """Load cleaned data files"""
    print("=" * 60)
    print("Loading data files...")
    print("=" * 60)
    
    # Load hall calls
    hall_calls_path = os.path.join(DATA_DIR, "clean_hall_calls.csv")
    hall_calls = pd.read_csv(hall_calls_path)
    print(f"Hall calls loaded: {len(hall_calls)} records")
    
    # Load car stops
    car_stops_path = os.path.join(DATA_DIR, "clean_car_stops.csv")
    car_stops = pd.read_csv(car_stops_path)
    print(f"Car stops loaded: {len(car_stops)} records")
    
    # Load load changes
    load_changes_path = os.path.join(DATA_DIR, "clean_load_changes.csv")
    load_changes = pd.read_csv(load_changes_path)
    print(f"Load changes loaded: {len(load_changes)} records")
    
    return hall_calls, car_stops, load_changes

def extract_system_parameters(hall_calls, car_stops):
    """Extract basic system parameters"""
    print("\n" + "=" * 60)
    print("Extracting system parameters...")
    print("=" * 60)
    
    # Get elevator IDs - handle different column naming conventions
    elevator_col = 'Elevator ID' if 'Elevator ID' in car_stops.columns else 'elevator_id'
    elevator_ids = sorted(car_stops[elevator_col].unique().tolist())
    print(f"Elevator IDs: {elevator_ids}")
    print(f"Number of elevators: {len(elevator_ids)}")
    
    # Get floor range - ensure integer conversion
    all_floors = []
    for col in ['Floor', 'floor', 'from_floor', 'to_floor']:
        if col in hall_calls.columns:
            floors = pd.to_numeric(hall_calls[col], errors='coerce').dropna()
            all_floors.extend([int(f) for f in floors])
    for col in ['Floor', 'floor']:
        if col in car_stops.columns:
            floors = pd.to_numeric(car_stops[col], errors='coerce').dropna()
            all_floors.extend([int(f) for f in floors])
    
    min_floor = int(min(all_floors)) if all_floors else 1
    max_floor = int(max(all_floors)) if all_floors else 14
    
    print(f"Floor range: {min_floor} - {max_floor}")
    
    # Calculate average wait time from hall calls
    if 'wait_time' in hall_calls.columns:
        avg_wait = hall_calls['wait_time'].mean()
        print(f"Average wait time: {avg_wait:.2f} seconds")
    else:
        avg_wait = 30.0  # Default value
    
    # Estimate travel time per floor (from car stops timestamps)
    avg_travel_time = 3.0  # Default: 3 seconds per floor
    
    params = {
        "elevator_ids": elevator_ids,
        "num_elevators": len(elevator_ids),
        "min_floor": min_floor,
        "max_floor": max_floor,
        "num_floors": max_floor - min_floor + 1,
        "avg_wait_time": float(avg_wait),
        "avg_travel_time_per_floor": avg_travel_time,
        "door_open_time": 4.0,
        "door_close_time": 2.0,
        "passenger_boarding_time": 2.0
    }
    
    # Save parameters
    with open(os.path.join(OUTPUT_DIR, "system_params.json"), 'w') as f:
        json.dump(params, f, indent=2)
    print(f"System parameters saved to {OUTPUT_DIR}/system_params.json")
    
    return params

def analyze_floor_demand(hall_calls, params):
    """Analyze demand patterns by floor and time"""
    print("\n" + "=" * 60)
    print("Analyzing floor demand patterns...")
    print("=" * 60)
    
    # Handle column naming
    time_col = 'Time' if 'Time' in hall_calls.columns else 'timestamp'
    floor_col = 'Floor' if 'Floor' in hall_calls.columns else 'from_floor'
    
    # Parse timestamp
    hall_calls['timestamp'] = pd.to_datetime(hall_calls[time_col])
    hall_calls['hour'] = hall_calls['timestamp'].dt.hour
    
    # Ensure floor is integer
    hall_calls['from_floor'] = pd.to_numeric(hall_calls[floor_col], errors='coerce')
    hall_calls = hall_calls.dropna(subset=['from_floor'])
    hall_calls['from_floor'] = hall_calls['from_floor'].astype(int)
    
    # Hourly demand by floor
    hourly_demand = hall_calls.groupby(['hour', 'from_floor']).size().reset_index(name='call_count')
    
    # Create pivot table
    demand_matrix = hourly_demand.pivot(index='from_floor', columns='hour', values='call_count').fillna(0)
    
    # Save hourly demand
    hourly_demand.to_csv(os.path.join(OUTPUT_DIR, "hourly_floor_demand.csv"), index=False)
    demand_matrix.to_csv(os.path.join(OUTPUT_DIR, "demand_matrix.csv"))
    
    print(f"Hourly demand analysis saved")
    print(f"Peak hours by floor:")
    for floor in range(params['min_floor'], params['max_floor'] + 1):
        if floor in demand_matrix.index:
            peak_hour = demand_matrix.loc[floor].idxmax()
            peak_count = demand_matrix.loc[floor].max()
            print(f"  Floor {floor}: peak at hour {peak_hour} ({int(peak_count)} calls)")
    
    return hourly_demand, demand_matrix

def analyze_floor_call_distribution(hall_calls, params):
    """Analyze call distribution by floor"""
    print("\n" + "=" * 60)
    print("Analyzing floor call distribution...")
    print("=" * 60)
    
    # Handle column naming
    floor_col = 'Floor' if 'Floor' in hall_calls.columns else 'from_floor'
    
    # Ensure floor is integer
    hall_calls['from_floor'] = pd.to_numeric(hall_calls[floor_col], errors='coerce').astype('Int64')
    hall_calls = hall_calls.dropna(subset=['from_floor'])
    
    floor_calls = hall_calls['from_floor'].value_counts().sort_index()
    floor_dist = (floor_calls / floor_calls.sum() * 100).round(2)
    
    # Save distribution
    dist_df = pd.DataFrame({
        'floor': floor_calls.index,
        'call_count': floor_calls.values,
        'percentage': floor_dist.values
    })
    dist_df.to_csv(os.path.join(OUTPUT_DIR, "floor_call_distribution.csv"), index=False)
    
    print("Floor call distribution:")
    for _, row in dist_df.iterrows():
        print(f"  Floor {int(row['floor'])}: {int(row['call_count'])} calls ({row['percentage']:.1f}%)")
    
    return dist_df

def analyze_wait_time_patterns(hall_calls, params):
    """Analyze wait time patterns"""
    print("\n" + "=" * 60)
    print("Analyzing wait time patterns...")
    print("=" * 60)
    
    if 'wait_time' not in hall_calls.columns:
        print("Warning: wait_time column not found, skipping wait time analysis")
        return None
    
    # Handle column naming
    time_col = 'Time' if 'Time' in hall_calls.columns else 'timestamp'
    floor_col = 'Floor' if 'Floor' in hall_calls.columns else 'from_floor'
    
    # Parse timestamp
    hall_calls['timestamp'] = pd.to_datetime(hall_calls[time_col])
    hall_calls['hour'] = hall_calls['timestamp'].dt.hour
    hall_calls['from_floor'] = pd.to_numeric(hall_calls[floor_col], errors='coerce').astype('Int64')
    
    # Wait time by hour
    hourly_wait = hall_calls.groupby('hour')['wait_time'].agg(['mean', 'std', 'max', 'count']).reset_index()
    hourly_wait.columns = ['hour', 'avg_wait', 'std_wait', 'max_wait', 'call_count']
    
    # Wait time by floor
    floor_wait = hall_calls.groupby('from_floor')['wait_time'].agg(['mean', 'std', 'max', 'count']).reset_index()
    floor_wait.columns = ['floor', 'avg_wait', 'std_wait', 'max_wait', 'call_count']
    
    # Long wait ratio (>60 seconds)
    long_wait_ratio = (hall_calls['wait_time'] > 60).mean() * 100
    
    # Save analysis
    hourly_wait.to_csv(os.path.join(OUTPUT_DIR, "hourly_wait_analysis.csv"), index=False)
    floor_wait.to_csv(os.path.join(OUTPUT_DIR, "floor_wait_analysis.csv"), index=False)
    
    print(f"Overall average wait time: {hall_calls['wait_time'].mean():.2f}s")
    print(f"Overall max wait time: {hall_calls['wait_time'].max():.2f}s")
    print(f"Long wait ratio (>60s): {long_wait_ratio:.2f}%")
    print(f"\nHourly wait time analysis saved")
    print(f"Floor wait time analysis saved")
    
    return hourly_wait, floor_wait

def analyze_idle_patterns(car_stops, load_changes, params):
    """Analyze elevator idle patterns"""
    print("\n" + "=" * 60)
    print("Analyzing elevator idle patterns...")
    print("=" * 60)
    
    # Handle column naming
    time_col = 'Time' if 'Time' in car_stops.columns else 'timestamp'
    floor_col = 'Floor' if 'Floor' in car_stops.columns else 'floor'
    elevator_col = 'Elevator ID' if 'Elevator ID' in car_stops.columns else 'elevator_id'
    
    # Parse timestamp
    car_stops['timestamp'] = pd.to_datetime(car_stops[time_col])
    car_stops['hour'] = car_stops['timestamp'].dt.hour
    car_stops['floor'] = pd.to_numeric(car_stops[floor_col], errors='coerce').astype('Int64')
    car_stops['elevator_id'] = car_stops[elevator_col]
    
    # Stops by elevator
    elevator_stops = car_stops.groupby('elevator_id').size().reset_index(name='stop_count')
    print("Elevator activity (stop count):")
    for _, row in elevator_stops.iterrows():
        print(f"  Elevator {row['elevator_id']}: {row['stop_count']} stops")
    
    # Floor popularity for parking
    floor_stops = car_stops.groupby('floor').size().reset_index(name='stop_count')
    floor_stops = floor_stops.sort_values('stop_count', ascending=False)
    
    print("\nTop parking floors (by stop frequency):")
    for _, row in floor_stops.head(5).iterrows():
        print(f"  Floor {int(row['floor'])}: {row['stop_count']} stops")
    
    # Save analysis
    elevator_stops.to_csv(os.path.join(OUTPUT_DIR, "elevator_activity.csv"), index=False)
    floor_stops.to_csv(os.path.join(OUTPUT_DIR, "floor_parking_frequency.csv"), index=False)
    
    return elevator_stops, floor_stops

def generate_parking_recommendations(hourly_demand, floor_dist, params):
    """Generate initial parking recommendations based on demand patterns"""
    print("\n" + "=" * 60)
    print("Generating parking recommendations...")
    print("=" * 60)
    
    # High demand floors (top 30% by call count)
    high_demand_floors = floor_dist.nlargest(int(len(floor_dist) * 0.3), 'call_count')['floor'].tolist()
    high_demand_floors = [int(f) for f in high_demand_floors]
    
    # Medium demand floors
    medium_demand_floors = floor_dist.nlargest(int(len(floor_dist) * 0.6), 'call_count')['floor'].tolist()
    medium_demand_floors = [int(f) for f in medium_demand_floors if f not in high_demand_floors]
    
    # Recommended parking strategy
    recommendations = {
        "high_demand_floors": high_demand_floors,
        "medium_demand_floors": medium_demand_floors,
        "recommended_parking_floors": high_demand_floors[:params['num_elevators']],
        "strategy_notes": [
            "Park elevators near high-demand floors during peak hours",
            "Distribute elevators evenly across demand zones during off-peak",
            "Ground floor (1) should always have at least 1 elevator during morning peak",
            "Consider zone-based parking for multi-elevator systems"
        ]
    }
    
    print(f"High demand floors: {high_demand_floors}")
    print(f"Medium demand floors: {medium_demand_floors}")
    print(f"Recommended initial parking positions: {recommendations['recommended_parking_floors']}")
    
    # Save recommendations
    with open(os.path.join(OUTPUT_DIR, "parking_recommendations.json"), 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    return recommendations

def main():
    print("=" * 60)
    print("Task 3: The Strategic Wait - Data Preprocessing")
    print("=" * 60)
    
    # Load data
    hall_calls, car_stops, load_changes = load_data()
    
    # Extract system parameters
    params = extract_system_parameters(hall_calls, car_stops)
    
    # Analyze floor demand patterns
    hourly_demand, demand_matrix = analyze_floor_demand(hall_calls.copy(), params)
    
    # Analyze floor call distribution
    floor_dist = analyze_floor_call_distribution(hall_calls.copy(), params)
    
    # Analyze wait time patterns
    wait_analysis = analyze_wait_time_patterns(hall_calls.copy(), params)
    
    # Analyze idle patterns
    idle_analysis = analyze_idle_patterns(car_stops.copy(), load_changes.copy(), params)
    
    # Generate parking recommendations
    recommendations = generate_parking_recommendations(hourly_demand, floor_dist, params)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Output files saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
