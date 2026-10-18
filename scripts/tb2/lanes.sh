# Serial lanes for tasks that interfere when run concurrently (containers share the host network and PID namespace).
# Tasks that conflict with each other MUST be in the same lane; lanes run in parallel, tasks inside a lane serially.
#   lane A: :8080 (configure-git-webserver, nginx-request-logging, pypi-server, install-windows-3.11) and qemu:
#           install-windows' verifier runs `pgrep qemu-system`, the two qemu tasks share :2222/:6665
#   lane B: :22/:8443 git-multibranch, :8000 headless-terminal, :5328 kv-store-grpc, and the two chrome tasks:
#           filter-js-from-html's verifier runs `pkill -f chrome`, which would hit break-filter-js-from-html
#   lane C: :29500 torch.distributed pair, :5000 hf-model-inference, :25 mailman, make-doom-for-mips, :6379 sanitize-git-repo
LANE_A="configure-git-webserver nginx-request-logging pypi-server install-windows-3.11 qemu-alpine-ssh qemu-startup"
LANE_B="git-multibranch headless-terminal kv-store-grpc filter-js-from-html break-filter-js-from-html"
LANE_C="torch-pipeline-parallelism torch-tensor-parallelism hf-model-inference mailman make-doom-for-mips sanitize-git-repo"
