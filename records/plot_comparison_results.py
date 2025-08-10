#!/usr/bin/env python3
"""
Sketch Comparison Results Visualization

This script reads the sketch_comparison_results_with_cmins.csv file and generates
two comprehensive figures showing execution time trends across different parameters:

1. Figure 1: Execution Time vs k (sketch size) with n=10000 fixed
2. Figure 2: Execution Time vs n (set size) with k=128 fixed

Each figure shows execution time performance for three sketch methods:
- FastSimilaritySketch
- DatasketchMinHashSketch
- CMinHashSketch

Figures are saved as PNG files in the records directory, focusing on execution time performance.

Time Complexity Analysis:
- FastSimilaritySketch: O(n + k log k) (expectation)
- DatasketchMinHashSketch: O(k * n)
- CMinHashSketch: O(k * n)

Author: PhD CS Student - Algorithmic Research
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for potential imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class ComparisonResultsPlotter:
    """
    Visualizes performance comparison results across different sketch algorithms.
    """
    
    def __init__(self, csv_filename: str = "sketch_comparison_results_simd.csv"):
        """
        Initialize the plotter with CSV file path.
        
        Args:
            csv_filename: Name of the CSV file containing comparison results
        """
        self.csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
        self.results_df = None
        self.load_data()
        
    def load_data(self) -> None:
        """
        Load comparison results from CSV file.
        """
        try:
            self.results_df = pd.read_csv(self.csv_path)
            print(f"Successfully loaded data from {self.csv_path}")
            print(f"Data shape: {self.results_df.shape}")
        except FileNotFoundError:
            print(f"Error: Could not find {self.csv_path}")
            print("Please run the comparison test first to generate the results file.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def plot_k_trends(self, fixed_n: int = 10000) -> None:
        """
        Plot execution time trends vs k (sketch size) with fixed n (set size).
        
        Args:
            fixed_n: Fixed set size for comparison
        """
        # Filter data for fixed n
        filtered_data = self.results_df[self.results_df['n'] == fixed_n].copy()
        
        if filtered_data.empty:
            print(f"Warning: No data found for n={fixed_n}")
            return
            
        # Sort by k for proper line plotting
        filtered_data = filtered_data.sort_values('k')
        
        # Create single figure for execution time
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'Sketch Algorithm Execution Time vs k (Set Size n={fixed_n})', fontsize=16, fontweight='bold')
        
        # Define colors and markers for each algorithm (KMin removed)
        colors = {
            'fast': '#2E86AB',      # Blue
            'datasketch': '#F18F01', # Orange
            'cmins': '#C73E1D'      # Red
        }
        
        markers = {
            'fast': 'o',
            'datasketch': '^',
            'cmins': 'D'
        }
        
        # Plot Execution Time vs k
        ax.plot(filtered_data['k'], filtered_data['fast_avg_time'], 
               color=colors['fast'], marker=markers['fast'], linewidth=3, markersize=8,
               label='FastSimilaritySketch', markeredgecolor='white', markeredgewidth=1)
        # KMinSketch line removed
        ax.plot(filtered_data['k'], filtered_data['datasketch_avg_time'], 
               color=colors['datasketch'], marker=markers['datasketch'], linewidth=3, markersize=8,
               label='DatasketchMinHash', markeredgecolor='white', markeredgewidth=1)
        ax.plot(filtered_data['k'], filtered_data['cmins_avg_time'], 
               color=colors['cmins'], marker=markers['cmins'], linewidth=3, markersize=8,
               label='CMinHashSketch', markeredgecolor='white', markeredgewidth=1)
        
        ax.set_xlabel('k (Sketch Size)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Execution Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('Execution Time vs Sketch Size', fontsize=16, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization of time differences
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(os.path.dirname(__file__), f'sketch_execution_time_vs_k_n{fixed_n}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved: {output_path}")
        plt.show()
    
    def plot_n_trends(self, fixed_k: int = 128) -> None:
        """
        Plot execution time trends vs n (set size) with fixed k (sketch size).
        
        Args:
            fixed_k: Fixed sketch size for comparison
        """
        # Filter data for fixed k
        filtered_data = self.results_df[self.results_df['k'] == fixed_k].copy()
        
        if filtered_data.empty:
            print(f"Warning: No data found for k={fixed_k}")
            return
            
        # Sort by n for proper line plotting
        filtered_data = filtered_data.sort_values('n')
        
        # Create single figure for execution time
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'Sketch Algorithm Execution Time vs n (Sketch Size k={fixed_k})', fontsize=16, fontweight='bold')
        
        # Define colors and markers for each algorithm (KMin removed)
        colors = {
            'fast': '#2E86AB',      # Blue
            'datasketch': '#F18F01', # Orange
            'cmins': '#C73E1D'      # Red
        }
        
        markers = {
            'fast': 'o',
            'datasketch': '^',
            'cmins': 'D'
        }
        
        # Plot Execution Time vs n
        ax.plot(filtered_data['n'], filtered_data['fast_avg_time'], 
               color=colors['fast'], marker=markers['fast'], linewidth=3, markersize=8,
               label='FastSimilaritySketch', markeredgecolor='white', markeredgewidth=1)
        # KMinSketch line removed
        ax.plot(filtered_data['n'], filtered_data['datasketch_avg_time'], 
               color=colors['datasketch'], marker=markers['datasketch'], linewidth=3, markersize=8,
               label='DatasketchMinHash', markeredgecolor='white', markeredgewidth=1)
        ax.plot(filtered_data['n'], filtered_data['cmins_avg_time'], 
               color=colors['cmins'], marker=markers['cmins'], linewidth=3, markersize=8,
               label='CMinHashSketch', markeredgecolor='white', markeredgewidth=1)
        
        ax.set_xlabel('n (Set Size)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Execution Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title('Execution Time vs Set Size', fontsize=16, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')  # Log scale for set size
        ax.set_yscale('log')  # Log scale for time
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(os.path.dirname(__file__), f'sketch_execution_time_vs_n_k{fixed_k}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved: {output_path}")
        plt.show()
    
    def generate_summary_stats(self) -> None:
        """
        Print summary statistics from the comparison results.
        """
        if self.results_df is None:
            print("No data loaded.")
            return
            
        print("\n" + "="*60)
        print("SKETCH ALGORITHM COMPARISON SUMMARY")
        print("="*60)
        
        # Overall averages
        print(f"\nOverall Average Execution Times:")
        print(f"  FastSimilaritySketch: {self.results_df['fast_avg_time'].mean():.6f}s")
        # KMinSketch removed
        print(f"  DatasketchMinHash:   {self.results_df['datasketch_avg_time'].mean():.6f}s")
        print(f"  CMinHashSketch:      {self.results_df['cmins_avg_time'].mean():.6f}s")
        
        print(f"\nOverall Average Estimation Errors:")
        print(f"  FastSimilaritySketch: {self.results_df['fast_avg_error'].mean():.6f}")
        # KMinSketch removed
        print(f"  DatasketchMinHash:   {self.results_df['datasketch_avg_error'].mean():.6f}")
        print(f"  CMinHashSketch:      {self.results_df['cmins_avg_error'].mean():.6f}")
        
        print(f"\nTesting Parameters:")
        print(f"  k values tested: {sorted(self.results_df['k'].unique())}")
        print(f"  n values tested: {sorted(self.results_df['n'].unique())}")
        print(f"  Total test configurations: {len(self.results_df)}")


def main():
    """
    Main function to generate all plots and statistics.
    """
    print("Sketch Algorithm Performance Visualization")
    print("==========================================")
    
    # Initialize plotter
    plotter = ComparisonResultsPlotter()
    
    # Generate summary statistics
    plotter.generate_summary_stats()
    
    # Generate plots
    print(f"\nGenerating performance plots...")
    plotter.plot_k_trends(fixed_n=10000)
    plotter.plot_n_trends(fixed_k=128)
    
    print(f"\nVisualization complete! Figures saved in records directory.")


if __name__ == '__main__':
    main() 