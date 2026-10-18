```bash
#!<abs_path>
#SBATCH -N 1
#SBATCH -J analysis
#SBATCH -o analysis.out
#SBATCH -e analysis.err
#SBATCH -t 00:30:00

module load python/3.9.13

python analysis.py
```