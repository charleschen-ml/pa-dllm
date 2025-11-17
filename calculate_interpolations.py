#!/usr/bin/env python3
"""
Generate interpolated manual_settings to cover all token positions.

Given a greedy block_size schedule like [2, 3, 3], generate manual_settings
that force intermediate block sizes to create training samples at positions
not covered by the greedy search.
"""

import csv
from typing import List, Dict


def calculate_interpolations(block_size: List[int]) -> List[Dict[int, int]]:
    """
    Generate interpolated manual_settings to cover intermediate positions.
    
    Strategy: For each block in the greedy schedule, try all smaller values (1 to size-1),
    keeping all previous blocks at their greedy values.
    
    Args:
        block_size: Original greedy schedule, e.g., [2, 3, 3]
    
    Returns:
        List of manual_settings dicts, where each dict maps block_index -> forced_size
        
    Example:
        >>> calculate_interpolations([2, 3, 3])
        [
            {0: 1},              # Decrement first block to 1
            {0: 2, 1: 1},        # Keep first, decrement second to 1
            {0: 2, 1: 2},        # Keep first, decrement second to 2
            {0: 2, 1: 3, 2: 1},  # Keep first two, decrement third to 1
            {0: 2, 1: 3, 2: 2},  # Keep first two, decrement third to 2
        ]
        
        >>> calculate_interpolations([3, 5])
        [
            {0: 1},          # Decrement first block to 1
            {0: 2},          # Decrement first block to 2
            {0: 3, 1: 1},    # Keep first, decrement second to 1
            {0: 3, 1: 2},    # Keep first, decrement second to 2
            {0: 3, 1: 3},    # Keep first, decrement second to 3
            {0: 3, 1: 4},    # Keep first, decrement second to 4
        ]
    
    Note: Greedy baseline is NOT included, as it's already in the CSV.
    """
    interpolations = []
    
    # NOTE: We skip the greedy baseline since it's already been generated
    
    # 2. For each block, try all smaller values (1 to block_size-1)
    for i, block_val in enumerate(block_size):
        # Try decrementing this block to values 1, 2, ..., block_val-1
        for new_val in range(1, block_val):
            # Create manual_settings that keeps blocks 0 to i-1 and sets block i to new_val
            manual_settings = {}
            
            # Keep all previous blocks at their greedy values
            for j in range(i):
                manual_settings[j] = block_size[j]
            
            # Set current block to the decremented value
            manual_settings[i] = new_val
            
            interpolations.append(manual_settings)
    
    return interpolations


def load_block_sizes_from_csv(csv_path: str) -> List[List[int]]:
    """
    Load block_size schedules from CSV file.
    
    Args:
        csv_path: Path to CSV file with 'block_size' column
    
    Returns:
        List of block_size schedules (parsed from string like '[2, 3, 3]')
    """
    block_sizes = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse block_size from string like '[2, 3, 3]' to list
            block_size_str = row['block_size'].strip('[]')
            if block_size_str:  # Handle empty strings
                block_size = [int(x.strip()) for x in block_size_str.split(',')]
                block_sizes.append(block_size)
    
    return block_sizes


def main():
    """Test the interpolation generation."""
    print("="*70)
    print("EXAMPLE 1: block_size=[2, 3, 3]")
    print("="*70)
    
    block_size = [2, 3, 3]
    interpolations = calculate_interpolations(block_size)
    
    print(f"Original greedy: {block_size} (total={sum(block_size)} tokens)")
    print(f"  → Greedy samples at positions: [0, 2, 5] (already in CSV)\n")
    print(f"Generated {len(interpolations)} interpolated manual_settings:\n")
    
    for i, manual_settings in enumerate(interpolations):
        # Calculate the position where this sample will be generated
        pos = sum(manual_settings.values())
        
        print(f"  {i+1}. manual_settings={manual_settings}")
        print(f"     → sample at position {pos}")
        print()
    
    print("\n" + "="*70)
    print("EXAMPLE 2: block_size=[3, 5]")
    print("="*70)
    
    block_size = [3, 5]
    interpolations = calculate_interpolations(block_size)
    
    print(f"Original greedy: {block_size} (total={sum(block_size)} tokens)")
    print(f"  → Greedy samples at positions: [0, 3] (already in CSV)\n")
    print(f"Generated {len(interpolations)} interpolated manual_settings:\n")
    
    for i, manual_settings in enumerate(interpolations):
        # Calculate the position where this sample will be generated
        pos = sum(manual_settings.values())
        
        print(f"  {i+1}. manual_settings={manual_settings}")
        print(f"     → sample at position {pos}")
        print()


if __name__ == '__main__':
    main()

