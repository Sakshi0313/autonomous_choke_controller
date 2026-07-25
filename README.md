\# Autonomous Production Choke Controller



\## Introduction

An autonomous choke controller solution for a single naturally flowing oil well. This method requires experience with Model Predictive Control (MPC) visualizing the well states, it uses scipy SLSQP optimization to maximize safe oil production whilst having all pressure constraints honored. 



\## Problem Statement

Develop a autonomous controller to tune an oil well choke position for target oil production while ensuring that WHP, FLP and BHP stay within constraints. 



\## Approach 

1\. Physics-based well simulator (Darcy's Law, Choke Flow Equation, Darcy-Weisbach) 

2\. System Identification 

3\. Open loop Step testing Grey-Box Predictive Model (quadratic steady-state + first order dynamics) 

4\. MPC controller using scipy SLSQP optimization and brute-force fallback Handles constraints with detection and attribution of infeasibility 



\## Tech stack 

\- Python 3.10+ 

\- NumPy, SciPy 

\- Matplotlib 

\- Jupyter Notebooks 



\~Sakshi Pawar

