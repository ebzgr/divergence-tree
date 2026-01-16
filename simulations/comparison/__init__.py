"""
Comparison simulation package.

This package contains tools for generating synthetic data and comparing
divergence tree methods with alternative approaches.
"""

from .data_generator import generate_comparison_data, get_data_summary

__all__ = [
    "generate_comparison_data",
    "get_data_summary",
]
