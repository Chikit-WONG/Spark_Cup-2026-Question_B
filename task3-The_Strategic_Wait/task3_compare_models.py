#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 3: The Strategic Wait - Model Comparison
Compares results from Model A (Q-Learning), Model B (NSGA-II), and Model C (Simulation)
All outputs are in English.
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

# Output directory
OUTPUT_DIR = "output/comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model output directories
MODEL_A_DIR = "output/model_A"
MODEL_B_DIR = "output/model_B"
MODEL_C_DIR = "output/model_C"


def load_model_results():
    """Load results from all three models"""
    results = {}
    
    # Model A: Q-Learning
    print("Loading Model A (Q-Learning) results...")
    model_a_path = os.path.join(MODEL_A_DIR, "evaluation_results.json")
    if os.path.exists(model_a_path):
        with open(model_a_path, 'r') as f:
            results['Model_A'] = json.load(f)
        print(f"  Loaded: AWT = {results['Model_A']['overall']['avg_wait']:.2f}s")
    else:
        print("  Warning: Model A results not found")
        results['Model_A'] = None
    
    # Model B: NSGA-II
    print("Loading Model B (NSGA-II) results...")
    model_b_path = os.path.join(MODEL_B_DIR, "best_solutions.json")
    if os.path.exists(model_b_path):
        with open(model_b_path, 'r') as f:
            results['Model_B'] = json.load(f)
        if 'best_awt' in results['Model_B']:
            awt = results['Model_B']['best_awt']['objectives'][0]
            print(f"  Loaded: Best AWT = {awt:.2f}s")
    else:
        print("  Warning: Model B results not found")
        results['Model_B'] = None
    
    # Model C: Simulation
    print("Loading Model C (Simulation) results...")
    model_c_path = os.path.join(MODEL_C_DIR, "simulation_summary.json")
    if os.path.exists(model_c_path):
        with open(model_c_path, 'r') as f:
            results['Model_C'] = json.load(f)
        awt = results['Model_C']['best_avg_wait_time']
        print(f"  Loaded: Best AWT = {awt:.2f}s")
    else:
        print("  Warning: Model C results not found")
        results['Model_C'] = None
    
    return results


def extract_key_metrics(results):
    """Extract key metrics from all models for comparison"""
    metrics = {}
    
    # Model A metrics
    if results['Model_A']:
        model_a = results['Model_A']
        metrics['Model_A'] = {
            'name': 'MDP + Q-Learning',
            'avg_wait_time': model_a['overall']['avg_wait'],
            'methodology': 'Reinforcement Learning',
            'best_strategy': 'Learned Policy',
            'key_insight': 'Adaptive strategy based on time and demand'
        }
    
    # Model B metrics
    if results['Model_B']:
        model_b = results['Model_B']
        if 'best_awt' in model_b:
            metrics['Model_B'] = {
                'name': 'NSGA-II Multi-Objective',
                'avg_wait_time': model_b['best_awt']['objectives'][0],
                'long_wait_ratio': model_b['best_awt']['objectives'][1],
                'energy': model_b['best_awt']['objectives'][2],
                'parking_floors': model_b['best_awt']['parking_floors'],
                'methodology': 'Evolutionary Algorithm',
                'key_insight': 'Pareto-optimal trade-offs between objectives'
            }
            
            if 'balanced' in model_b:
                metrics['Model_B']['balanced_solution'] = {
                    'avg_wait_time': model_b['balanced']['objectives'][0],
                    'parking_floors': model_b['balanced']['parking_floors']
                }
    
    # Model C metrics
    if results['Model_C']:
        model_c = results['Model_C']
        metrics['Model_C'] = {
            'name': 'Discrete Event Simulation',
            'avg_wait_time': model_c['best_avg_wait_time'],
            'best_strategy': model_c['best_overall_strategy'],
            'parking_floors': model_c['best_parking_floors'],
            'methodology': 'Simulation-based Validation',
            'key_insight': 'Empirical validation of strategies'
        }
        
        if 'all_strategies' in model_c:
            metrics['Model_C']['all_strategies'] = model_c['all_strategies']
    
    return metrics


def create_comparison_table(metrics):
    """Create a comparison table of all models"""
    print("\n" + "=" * 70)
    print("Model Comparison Summary")
    print("=" * 70)
    
    # Print comparison
    print(f"\n{'Model':<25} {'Methodology':<25} {'Best AWT (s)':<15}")
    print("-" * 65)
    
    for model, data in metrics.items():
        print(f"{data['name']:<25} {data['methodology']:<25} {data['avg_wait_time']:.2f}")
    
    # Create DataFrame
    comparison_data = []
    for model, data in metrics.items():
        row = {
            'Model': data['name'],
            'Methodology': data['methodology'],
            'Best_AWT_seconds': data['avg_wait_time'],
            'Key_Insight': data.get('key_insight', '')
        }
        if 'long_wait_ratio' in data:
            row['Long_Wait_Ratio'] = data['long_wait_ratio']
        if 'parking_floors' in data:
            row['Best_Parking_Floors'] = str(data['parking_floors'])
        if 'best_strategy' in data:
            row['Best_Strategy'] = data['best_strategy']
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


def visualize_comparison(metrics, results):
    """Create comparison visualizations"""
    print("\nGenerating comparison visualizations...")
    
    # 1. AWT Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bar chart of AWT
    ax1 = axes[0, 0]
    models = list(metrics.keys())
    awt_values = [metrics[m]['avg_wait_time'] for m in models]
    model_names = [metrics[m]['name'] for m in models]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax1.bar(model_names, awt_values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Average Wait Time (s)', fontsize=11)
    ax1.set_title('Model Comparison: Average Wait Time', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, awt_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Methodology comparison (radar chart style represented as grouped bar)
    ax2 = axes[0, 1]
    
    criteria = ['Speed', 'Adaptability', 'Optimality', 'Interpretability']
    model_scores = {
        'Model_A': [3, 5, 3, 4],  # Q-Learning: adaptive but slower to converge
        'Model_B': [4, 3, 5, 3],  # NSGA-II: optimal but less adaptive
        'Model_C': [5, 2, 4, 5]   # Simulation: fast, interpretable, good validation
    }
    
    x = np.arange(len(criteria))
    width = 0.25
    
    for i, (model, scores) in enumerate(model_scores.items()):
        if model in metrics:
            ax2.bar(x + i * width, scores, width, label=metrics[model]['name'],
                   color=colors[i], alpha=0.8)
    
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(criteria)
    ax2.set_ylabel('Score (1-5)')
    ax2.set_title('Model Characteristics Comparison', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 6)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Parking floor distribution comparison
    ax3 = axes[1, 0]
    
    y_pos = 0
    yticks = []
    yticklabels = []
    
    for model, data in metrics.items():
        if 'parking_floors' in data:
            floors = data['parking_floors']
            ax3.scatter(floors, [y_pos] * len(floors), s=100, alpha=0.7,
                       label=f"{data['name']}")
            yticks.append(y_pos)
            yticklabels.append(data['name'])
            y_pos += 1
    
    ax3.set_yticks(yticks)
    ax3.set_yticklabels(yticklabels)
    ax3.set_xlabel('Floor', fontsize=11)
    ax3.set_title('Best Parking Configuration by Model', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = """
    Key Findings:
    
    1. Model A (Q-Learning):
       - Learns adaptive strategies based on time and demand
       - Best for dynamic, changing conditions
       - Requires training time but generalizes well
    
    2. Model B (NSGA-II):
       - Finds Pareto-optimal solutions
       - Balances multiple objectives (AWT, LWR, Energy)
       - Best for understanding trade-offs
    
    3. Model C (Simulation):
       - Validates strategies with realistic scenarios
       - Provides interpretable results
       - Best for practical implementation testing
    
    Recommendation:
    Use Model B for initial strategy design, validate with Model C,
    and implement Model A for adaptive real-time control.
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/model_comparison.png")
    
    # 2. Strategy consistency analysis (if Model C has multiple strategies)
    if 'Model_C' in metrics and 'all_strategies' in metrics['Model_C']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategies = metrics['Model_C']['all_strategies']
        names = list(strategies.keys())
        awt_values = [strategies[n]['avg_wait_time'] for n in names]
        lwr_values = [strategies[n]['long_wait_ratio'] for n in names]
        
        # Sort by AWT
        sorted_idx = np.argsort(awt_values)
        names = [names[i] for i in sorted_idx]
        awt_values = [awt_values[i] for i in sorted_idx]
        lwr_values = [lwr_values[i] for i in sorted_idx]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, awt_values, width, label='Avg Wait Time (s)', 
                       color='steelblue', alpha=0.8)
        
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, lwr_values, width, label='Long Wait Ratio (%)',
                        color='coral', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Average Wait Time (s)', color='steelblue')
        ax2.set_ylabel('Long Wait Ratio (%)', color='coral')
        ax.set_title('Strategy Performance (from Simulation)', fontsize=12, fontweight='bold')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'strategy_performance.png'), dpi=150)
        plt.close()
        print(f"  Saved: {OUTPUT_DIR}/strategy_performance.png")


def generate_recommendations(metrics):
    """Generate final recommendations based on all models"""
    print("\n" + "=" * 70)
    print("Final Recommendations")
    print("=" * 70)
    
    recommendations = []
    
    # Find best overall
    best_model = min(metrics.items(), key=lambda x: x[1]['avg_wait_time'])
    
    recommendations.append({
        'category': 'Best Overall Strategy',
        'model': best_model[0],
        'details': f"Based on minimum AWT of {best_model[1]['avg_wait_time']:.2f}s",
        'parking_floors': best_model[1].get('parking_floors', 'See model output')
    })
    
    # For different scenarios
    if 'Model_A' in metrics:
        recommendations.append({
            'category': 'Dynamic/Adaptive Control',
            'model': 'Model_A',
            'details': 'Use Q-Learning policy for real-time adaptive parking',
            'use_case': 'Buildings with varying demand patterns'
        })
    
    if 'Model_B' in metrics and 'balanced_solution' in metrics['Model_B']:
        recommendations.append({
            'category': 'Balanced Multi-Objective',
            'model': 'Model_B',
            'details': 'Use balanced Pareto solution for trade-off optimization',
            'parking_floors': metrics['Model_B']['balanced_solution']['parking_floors']
        })
    
    if 'Model_C' in metrics:
        recommendations.append({
            'category': 'Practical Implementation',
            'model': 'Model_C',
            'details': f"Best validated strategy: {metrics['Model_C'].get('best_strategy', 'N/A')}",
            'parking_floors': metrics['Model_C'].get('parking_floors', 'See simulation output')
        })
    
    # Print recommendations
    for rec in recommendations:
        print(f"\n{rec['category']}:")
        print(f"  Model: {rec['model']}")
        print(f"  Details: {rec['details']}")
        if 'parking_floors' in rec:
            print(f"  Parking Floors: {rec['parking_floors']}")
    
    return recommendations


def save_comparison_results(metrics, comparison_df, recommendations):
    """Save all comparison results"""
    print("\nSaving comparison results...")
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, 'comparison_metrics.json'), 'w') as f:
        # Convert non-serializable items
        metrics_clean = {}
        for model, data in metrics.items():
            metrics_clean[model] = {
                k: (str(v) if isinstance(v, (list, dict)) else v)
                for k, v in data.items()
            }
        json.dump(metrics_clean, f, indent=2)
    
    # Save recommendations
    with open(os.path.join(OUTPUT_DIR, 'recommendations.json'), 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)
    
    print(f"  Results saved to {OUTPUT_DIR}/")


def main():
    print("=" * 70)
    print("Task 3: Model Comparison")
    print("=" * 70)
    
    # Load results from all models
    results = load_model_results()
    
    # Check if any results exist
    if all(v is None for v in results.values()):
        print("\nError: No model results found. Please run the individual models first:")
        print("  1. python task3_preprocess.py")
        print("  2. python task3_model_A_MDP_QLearning.py")
        print("  3. python task3_model_B_MultiObjective.py")
        print("  4. python task3_model_C_Simulation.py")
        return
    
    # Extract metrics
    metrics = extract_key_metrics(results)
    
    if not metrics:
        print("\nError: Could not extract metrics from model results.")
        return
    
    # Create comparison table
    comparison_df = create_comparison_table(metrics)
    
    # Visualize comparison
    visualize_comparison(metrics, results)
    
    # Generate recommendations
    recommendations = generate_recommendations(metrics)
    
    # Save results
    save_comparison_results(metrics, comparison_df, recommendations)
    
    print("\n" + "=" * 70)
    print("Model Comparison Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
