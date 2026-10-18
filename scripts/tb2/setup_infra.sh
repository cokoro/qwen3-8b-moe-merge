#!/bin/bash
set -x
export DEBIAN_FRONTEND=noninteractive
# apt -> tuna mirror
sed -i 's#http://archive.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g; s#http://security.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' /etc/apt/sources.list
apt-get update -y
apt-get install -y ca-certificates curl gnupg iptables tmux jq git
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu jammy stable" > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-buildx-plugin
# pip mirror
mkdir -p /root/.config/pip
printf "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\n" > /root/.config/pip/pip.conf
/opt/conda/bin/pip install -U uv
echo SETUP_INFRA_DONE
