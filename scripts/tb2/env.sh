# Source before running harbor on this host
export PATH=/root/gpufree-data/tb2/bin:/root/.local/bin:$PATH
export LITELLM_LOCAL_MODEL_COST_MAP=True   # raw.githubusercontent.com unreachable
export TB2_DATASET=/root/gpufree-data/datasets/tb2-mirror
export SIF_CACHE=/root/gpufree-data/sif-cache
export APPTAINER_CACHEDIR=/root/gpufree-data/apptainer-cache   # harbor-triggered image pulls: keep OCI blobs off the 30G root overlay
export APPTAINER_TMPDIR=/root/gpufree-data/apptainer-tmp
