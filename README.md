# Water Park Simulation

This project is a discrete-event simulation of a water park, developed as part of an academic course in simulation.

The goal of the project is to model operational behavior in a water park and analyze system performance under different configurations using simulation techniques.

---

## Project Overview

The simulation represents a water park environment with multiple activities, visitor groups, and capacity constraints.

The model simulates:
- Visitor arrivals and group behavior  
- Queue formation and waiting times  
- Activity capacity limitations  
- Time-based event handling  
- Visitor flow throughout the system  

The project focuses on analyzing congestion, system efficiency, and performance measures using controlled simulation experiments.

---

## Simulation Approach

The simulation is implemented using a discrete-event approach and an object-oriented design.

Different scenarios and system configurations are evaluated by:
- Running the simulation with different random seeds  
- Comparing performance metrics across multiple runs  
- Analyzing the impact of system parameters on waiting times and congestion  

This approach allows controlled comparison between alternatives while keeping the model structure fixed.

---

## Data File

The project includes an Excel file:

**samples for course project.xlsx**

This file contains descriptive data related to some of the park’s activities and was used to understand arrival characteristics and support modeling assumptions.

---

## How to Run

Make sure Python is installed on your system.

When running the program, you will be asked to select the Excel file containing the input data.

Then run the simulation using:

```bash
python water_park_simulation.py
