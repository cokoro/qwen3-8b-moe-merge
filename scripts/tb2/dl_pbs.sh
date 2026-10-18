#!/bin/bash
# fetch python-build-standalone builds used by uv 0.9.5 into the local mirror (parallel, resumable)
P=/root/gpufree-data/tb2/mirror/python-build-standalone
fetch() { u="$1"; rel=$(echo $u | cut -d/ -f8); f=$(echo $u | cut -d/ -f9); mkdir -p $P/$rel
  curl -sfL --retry 5 -C - -o "$P/$rel/$f.part" "https://registry.npmmirror.com/-/binary/python-build-standalone/$rel/$f" && mv "$P/$rel/$f.part" "$P/$rel/$f" && echo "OK $f" || echo "FAIL $f"; }
export -f fetch; export P
grep -E "3\.13|3\.12\.12.*musl" /tmp/pbs_urls.txt | xargs -P 3 -I{} bash -c "fetch {}"
echo PBS_DONE
