#!/bin/bash
# Pre-convert every TB2 task image to SIF, using harbor's cache naming (image with / and : -> _)
export APPTAINER_TMPDIR=/root/gpufree-data/apptainer-tmp
pull() {
  img="$1"; name=$(echo "$img" | tr "/:" "__")
  out="/root/gpufree-data/sif-cache/${name}.sif"
  [ -f "$out" ] && { echo "SKIP $img"; return; }
  if apptainer pull --disable-cache "${out}.tmp" "docker://$img" > "/root/gpufree-data/sif-cache/${name}.log" 2>&1; then
    mv "${out}.tmp" "$out"; echo "OK $img"
  else
    rm -f "/root/gpufree-data/sif-cache/${name}.sif.tmp"; echo "FAIL $img"
  fi
}
export -f pull
grep -h "^docker_image" /root/gpufree-data/datasets/tb2-mirror/*/task.toml | sed 's/.*= "//; s/"//' | sort -u | xargs -P 6 -I{} bash -c 'pull {}'
echo PREPULL_DONE
