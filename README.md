# Memristor Simulation Project

This repository contains simulations of memristor devices and neural network experiments using the generated data.

## Steps to Run

1. **Simulate Memristor**
   - Use the Verilog-A model file `veriloga.va` to simulate the memristor behavior in Cadence

2. **Export Simulation Data**
   - Export the results into a `.csv` file.
   - Include the following columns:
     - Current (`I`) — for example `/I9/p (A)`
     - Voltage (`V`) — for example `/net1 (V)`

3. **Run Jupyter Notebook**
   - Open the notebook file `Hystheresis.ipynb` files.
   - Update the file path and column names in the notebook to match your exported `.csv` data.
   - Execute the notebook to analyze and visualize the results.

## Notes

- Make sure the `.va` and `.ipynb` files are in the correct branch:
  - `memristor-cadence` for Verilog-A simulations
  - `neuralNetwork-sim` for neural network analysis
