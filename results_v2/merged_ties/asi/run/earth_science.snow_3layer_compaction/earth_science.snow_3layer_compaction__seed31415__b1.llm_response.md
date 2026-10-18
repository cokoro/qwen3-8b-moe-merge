```bash
#!<abs_path>
#SBATCH --job-name=snowpack
#SBATCH --output=snowpack.out
#SBATCH --error=snowpack.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --mem=5G

module load python/3.9
source activate snowpack

python analysis.py
```