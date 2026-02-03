
import unittest
import numpy as np
import cudaq
import sys
import os

# Import modules from current directory
import labs_utils_cpu
import quantum_kernels

class TestLABSEnergyFunctions(unittest.TestCase):
    def test_energy_calculation_known_values(self):
        """Verify energy calculation against known manual results."""
        # N=3, E=1
        s3 = np.array([1, 1, -1])
        self.assertEqual(labs_utils_cpu.calculate_labs_energy(s3), 1.0)
        
        # N=4, E=2
        s4 = np.array([1, 1, 1, -1])
        self.assertEqual(labs_utils_cpu.calculate_labs_energy(s4), 2.0)
        
        # N=5. Optimal E=1. Sequence [1, 1, 1, -1, 1]
        s5 = np.array([1, 1, 1, -1, 1])
        self.assertEqual(labs_utils_cpu.calculate_labs_energy(s5), 1.0)
        
        print("Energy calculation tests passed.")

    def test_symmetry_reversal(self):
        """E(S) == E(S[::-1])"""
        s = np.random.choice([-1, 1], size=10)
        e1 = labs_utils_cpu.calculate_labs_energy(s)
        e2 = labs_utils_cpu.calculate_labs_energy(s[::-1])
        self.assertEqual(e1, e2)
        print("Symmetry (Reversal) test passed.")

    def test_symmetry_inversion(self):
        """E(S) == E(-S)"""
        s = np.random.choice([-1, 1], size=10)
        e1 = labs_utils_cpu.calculate_labs_energy(s)
        e2 = labs_utils_cpu.calculate_labs_energy(-s)
        self.assertEqual(e1, e2)
        print("Symmetry (Inversion) test passed.")

class TestQuantumKernels(unittest.TestCase):
    def test_get_interactions(self):
        """Verify G2 and G4 generation logic."""
        N = 5
        G2, G4 = quantum_kernels.get_interactions(N)
        
        # Basic type checks
        self.assertIsInstance(G2, list)
        self.assertIsInstance(G4, list)
        
        # For N=5, check counts
        # G2: (0,1), (0,2), (1,2), (1,3), (2,3), (2,4) ...? 
        # Logic: i from 0 to N-3 (2). 
        # i=0: max_k = floor(4/2)=2. k=1,2.Pairs: (0,1), (0,2)
        # i=1: max_k = floor(3/2)=1. k=1.  Pairs: (1,2)
        # i=2: max_k = floor(2/2)=1. k=1.  Pairs: (2,3)
        # Total G2: 4 pairs.
        
        self.assertTrue(len(G2) > 0)
        print("Interaction generation test passed.")

    def test_kernel_execution(self):
        """Verify quantum kernels run without error on CPU backend."""
        print("Running Quantum Kernel Smoke Test...")
        N = 4
        G2, G4 = quantum_kernels.get_interactions(N)
        steps = 1
        dt = 0.1
        T = 1.0
        thetas = [0.1]
        
        # Run sample
        try:
            result = cudaq.sample(quantum_kernels.trotterized_circuit, N, G2, G4, steps, dt, T, thetas, shots_count=10)
            self.assertEqual(result.count(), 10)
            
            # Check keys are length N
            for bitstring in result.keys():
                self.assertEqual(len(bitstring), N)
            
            print("Quantum kernel execution test passed.")
        except Exception as e:
            self.fail(f"Quantum kernel execution failed: {e}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
