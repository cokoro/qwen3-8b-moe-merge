```bash
openmmrun -p 0 -n 4 -o dipeptide -f amber14-all.xml -s data/dipeptide.pdb -t 300 -i 300 -dt 2 -nsteps 50000000 -cv "phi:4,6,8,15" "psi:6,8,15,17"
```