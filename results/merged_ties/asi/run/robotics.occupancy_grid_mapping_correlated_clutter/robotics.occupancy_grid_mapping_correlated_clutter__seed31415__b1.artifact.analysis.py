```bash
#!/bin/bash
set -e
mkdir -p data
cd data
wget -O metadata.json 'https://www.dropbox.com/s/90z4x6s6y8q627b/metadata.json?dl=1'
wget -O poses.npy 'https://www.dropbox.com/s/3z62xk40l523775/poses.npy?dl=1'
wget -O ranges.npy 'https://www.dropbox.com/s/0431769k50x0q9k/ranges.npy?dl=1'
wget -O scan_angles.npy 'https://www.dropbox.com/s/20138z0k0v4203c/scan_angles.npy?dl=1'
cd ..
```