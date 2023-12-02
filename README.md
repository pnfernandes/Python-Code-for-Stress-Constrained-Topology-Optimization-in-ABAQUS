# Python-Code-for-Stress-Constrained-Topology-Optimization-in-ABAQUS
The file here included contains a Python code with five implementations of topology optimization approaches suitable for 2D and 3D problems, all considering bi-directional evolutionary structural optimization. The approaches implemented include both discrete and continuous methods, namely:
 - Optimality Criteria, for continuous or discrete variables;
 - Method of Moving Asymptotes;
 - Sequential Least-Squares Programming (from SciPy module);
 - Trust-region (from SciPy module).

The implementation of the Optimality Criteria method is suitable for compliance minimization problems with one mass or volume constraint. The implementation of the remaining methods is suitable for stress constrained compliance minimization and stress minimization problems, both with one mass or volume constraint.

The code uses the commercial software ABAQUS to execute Finite Element Analysis (FEA) and automatically access most of the necessary information for the optimization process, such as initial design, material properties, and loading conditions from a model database file (.cae) while providing a simple graphic user interface. Although the code has been developed mainly for educational purposes, its modularity allows for easy editing and extension to other topology optimization problems, making it interesting for more experienced researchers.

This code has been used in the article "Stress constrained topology optimization for commercial software: a Python implementation for ABAQUS®" [1].
Furthermore, a Mendeley Dataset [2] provides access to the results obtained, as well as the information necessary to replicate them.

Notes:
- Stress-dependent problems are only compatible with the following ABAQUS element types: CPE4, CPS4, and 3DQ8.
- The authorship of the functions 'mmasub' and 'subsolv' used in the Method of Moving Asymptotes are credited to Arjen Deetman. Source: https://github.com/arjendeetman/GCMMA-MMA-Python
- Despite the validations performed, this program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# Licence: CC BY 4.0
You can share, copy and modify this dataset so long as you give appropriate credit, provide a link to the CC BY license, and indicate if changes were made, but you may not do so in a way that suggests the rights holder has endorsed you or your use of the dataset. Note that further permission may be required for any content within the dataset that is identified as belonging to a third party.

# References:
[1] - P. Fernandes et al., (2023) Stress-Constrained Topology Optimization for Commercial Software: A Python Implementation for ABAQUS®. Appl. Sci. 13, 12916. doi: 10.3390/app132312916

[2] - P. Fernandes et al., (2023) Python Code for Stress Constrained Topology Optimization in ABAQUS, Mendeley Data, V1, doi: 10.17632/d347zjsk27.1
