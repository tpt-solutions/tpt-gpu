#!/usr/bin/env python3
"""
JAX interop example for tptr framework backends.

NOTE: JAX integration is not yet implemented. This example will print a
not-implemented notice and exit. See layer6_framework/examples/basic_usage.py
for a working example using the PyTorch backend or the raw tptr Python API.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tptr.jax


def main():
    print("=" * 60)
    print("TPT Framework Backends - JAX Interop Example")
    print("=" * 60)

    if not tptr.jax.is_available():
        print(
            "\nJAX integration is not yet implemented.\n"
            "tptr.jax.is_available() returned False.\n"
            "Use the PyTorch backend or the raw tptr Python API instead."
        )
        return

    print("\nJAX is available — backend integration code goes here.")


if __name__ == "__main__":
    main()

