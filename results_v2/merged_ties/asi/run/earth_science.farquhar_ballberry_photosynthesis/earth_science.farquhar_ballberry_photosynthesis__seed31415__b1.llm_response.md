```bash
#!<abs_path>

set -e

# This script is run from the project root directory.
# It does the following:
# - Runs the analysis on the provided data files
# - Saves the outputs in the results directory
# - Plots the outputs

# Load Python module
source activate python3

# Run the analysis
python analysis.py

# Plot the outputs
python plot.py

# Exit
exit 0
```