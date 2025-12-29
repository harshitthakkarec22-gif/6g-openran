# Getting Started with 6G OPENRAN

This guide will help you get started with the 6G OPENRAN project, from setting up your environment to running your first simulation.

## Prerequisites

### Hardware Requirements

**Minimum Configuration**:
- CPU: Intel Core i5/i7 or AMD Ryzen 5/7 (8+ cores)
- RAM: 16GB
- Storage: 100GB SSD
- Network: 1Gbps Ethernet

**Recommended Configuration**:
- CPU: Intel Core i9 or AMD Ryzen 9 (16+ cores)
- RAM: 32GB or more
- Storage: 256GB NVMe SSD
- GPU: NVIDIA RTX 3060+ (for ML workloads)
- Network: 10Gbps Ethernet

**Optional Hardware**:
- USRP (Ettus B210/X310) for real RF testing
- SDR (LimeSDR, HackRF) for spectrum analysis

### Software Requirements

**Operating System**:
- Ubuntu 20.04 LTS or 22.04 LTS (strongly recommended)
- Other Linux distributions may work but are not officially supported

**Essential Tools**:
```bash
# Core development tools
- Git 2.30+
- Python 3.8+
- GCC/G++ 9.0+
- CMake 3.16+

# Container tools
- Docker 20.10+
- Docker Compose 1.29+
- Kubernetes 1.24+ (minikube or kind for local)

# Build tools
- make
- autotools
- pkg-config
```

## Step-by-Step Setup

### Step 1: System Preparation

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install essential build tools
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    net-tools \
    iputils-ping \
    tcpdump \
    wireshark

# Install development libraries
sudo apt-get install -y \
    libfftw3-dev \
    libmbedtls-dev \
    libboost-program-options-dev \
    libconfig++-dev \
    libsctp-dev \
    libyaml-cpp-dev \
    libzmq3-dev
```

### Step 2: Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group
sudo usermod -aG docker $USER

# Logout and login for group changes to take effect
# Or run: newgrp docker

# Verify Docker installation
docker --version
docker run hello-world

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify Docker Compose
docker-compose --version
```

### Step 3: Install Kubernetes (minikube)

```bash
# Download and install minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Verify installations
minikube version
kubectl version --client

# Start minikube (allocate sufficient resources)
minikube start --cpus=4 --memory=8192 --disk-size=50g

# Enable required addons
minikube addons enable metrics-server
minikube addons enable dashboard
```

### Step 4: Install Python and Virtual Environment

```bash
# Install Python 3.8+ (if not already installed)
sudo apt-get install -y python3 python3-pip python3-venv

# Verify Python version
python3 --version

# Create virtual environment for the project
python3 -m venv ~/venv/6g-openran

# Activate virtual environment
source ~/venv/6g-openran/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install common Python packages
pip install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    jupyter \
    pytest \
    requests \
    pyyaml
```

### Step 5: Clone the Repository

```bash
# Navigate to your projects directory
cd ~/projects  # or wherever you keep your projects

# Clone the repository
git clone https://github.com/harshitthakkarec22-gif/6g-openran.git
cd 6g-openran

# Check the repository structure
ls -la

# Read the main README
cat README.md
```

### Step 6: Install Project Dependencies

```bash
# Navigate to project root
cd ~/projects/6g-openran

# Make scripts executable
chmod +x scripts/**/*.sh

# Run environment check script (coming soon)
# ./scripts/utils/check-environment.sh

# Install additional dependencies as needed
# Follow component-specific installation guides
```

## Setting Up Development Tools

### Install Open Source 5G Tools

#### srsRAN (Software Radio Systems RAN)

```bash
# Install dependencies
sudo apt-get install -y \
    libfftw3-dev \
    libmbedtls-dev \
    libboost-program-options-dev \
    libconfig++-dev \
    libsctp-dev

# Clone srsRAN
cd ~/projects
git clone https://github.com/srsran/srsRAN_4G.git
cd srsRAN_4G

# Build srsRAN
mkdir build && cd build
cmake ../
make -j$(nproc)
sudo make install
sudo ldconfig

# Verify installation
srsepc --version
srsenb --version
srsue --version
```

#### Open5GS (5G Core Network)

```bash
# Add Open5GS repository
sudo add-apt-repository ppa:open5gs/latest
sudo apt-get update

# Install Open5GS
sudo apt-get install -y open5gs

# Verify installation
systemctl status open5gs-amfd
systemctl status open5gs-smfd
systemctl status open5gs-upfd

# Configuration files location
ls -la /etc/open5gs/
```

#### ns-3 Network Simulator

```bash
# Install dependencies
sudo apt-get install -y \
    g++ \
    python3 \
    python3-dev \
    pkg-config \
    sqlite3 \
    libsqlite3-dev \
    libxml2 \
    libxml2-dev \
    libgtk-3-dev \
    gir1.2-goocanvas-2.0 \
    python3-gi \
    python3-gi-cairo \
    python3-pygraphviz \
    python3-setuptools

# Download ns-3
cd ~/projects
wget https://www.nsnam.org/releases/ns-allinone-3.39.tar.bz2
tar xjf ns-allinone-3.39.tar.bz2
cd ns-allinone-3.39/ns-3.39

# Configure and build
./ns3 configure --enable-examples --enable-tests
./ns3 build

# Verify installation
./ns3 run first
```

### IDE Setup

#### Visual Studio Code (Recommended)

```bash
# Install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt-get update
sudo apt-get install -y code

# Launch VS Code
code ~/projects/6g-openran
```

**Recommended Extensions**:
- Python (Microsoft)
- C/C++ (Microsoft)
- Go
- Docker
- Kubernetes
- GitLens
- YAML
- Markdown All in One

## First Steps

### 1. Explore the Documentation

```bash
# Read the comprehensive research report
cd ~/projects/6g-openran
cat docs/RESEARCH_REPORT.md | less

# Review the architecture
cat docs/architecture/INITIAL_ARCHITECTURE.md | less

# Check component READMEs
find . -name "README.md" -type f
```

### 2. Verify Environment

```bash
# Check Docker
docker ps
docker images

# Check Kubernetes
kubectl cluster-info
kubectl get nodes

# Check Python environment
python3 --version
pip list

# Check installed tools
srsepc --version
open5gs-amfd --version
```

### 3. Run Your First Test

```bash
# Start srsRAN EPC in one terminal
sudo srsepc

# In another terminal, start srsRAN eNodeB
sudo srsenb

# In a third terminal, start srsRAN UE
sudo srsue

# Observe the connection establishment
# Check interfaces: ip addr show
```

### 4. Deploy Basic Components (Coming Soon)

```bash
# Deploy RAN components
kubectl apply -f configs/ran/

# Deploy Core Network
kubectl apply -f configs/core/

# Check deployment status
kubectl get pods
kubectl get services
```

## Understanding the Architecture

### Key Concepts to Learn

1. **RAN Disaggregation**:
   - Centralized Unit (CU): Control and user plane separation
   - Distributed Unit (DU): MAC/RLC processing
   - Radio Unit (RU): RF processing

2. **5G Core Network**:
   - Control Plane: AMF, SMF, PCF, AUSF, UDM
   - User Plane: UPF
   - Service-based architecture (SBA)

3. **O-RAN Interfaces**:
   - A1: Non-RT RIC ↔ Near-RT RIC
   - E2: Near-RT RIC ↔ RAN
   - O1: Management interface
   - Open Fronthaul: DU ↔ RU

4. **RAN Intelligent Controller (RIC)**:
   - Near-RT RIC: Real-time control (10ms-1s)
   - Non-RT RIC: Policy and ML model management (>1s)
   - xApps: RIC applications

### Recommended Learning Path

**Week 1-2: Fundamentals**
- [ ] Read Research Report sections 1-3
- [ ] Understand basic 5G architecture
- [ ] Learn O-RAN principles
- [ ] Set up development environment

**Week 3-4: Hands-On with Tools**
- [ ] Run srsRAN examples
- [ ] Deploy Open5GS core
- [ ] Test end-to-end connectivity
- [ ] Capture and analyze packets with Wireshark

**Week 5-6: Deep Dive into Components**
- [ ] Study RAN protocol stack
- [ ] Understand core network functions
- [ ] Learn about network slicing
- [ ] Experiment with configurations

**Week 7-8: Advanced Topics**
- [ ] Implement basic xApp
- [ ] Test RIC functionality
- [ ] Explore AI/ML integration
- [ ] Study 6G-specific features

## Common Issues and Solutions

### Issue: Docker permission denied

```bash
# Solution: Add user to docker group
sudo usermod -aG docker $USER
# Then logout and login, or run:
newgrp docker
```

### Issue: Kubernetes not starting

```bash
# Solution: Delete and restart minikube
minikube delete
minikube start --cpus=4 --memory=8192
```

### Issue: srsRAN build errors

```bash
# Solution: Install missing dependencies
sudo apt-get install -y \
    build-essential \
    cmake \
    libfftw3-dev \
    libmbedtls-dev \
    libboost-program-options-dev \
    libconfig++-dev \
    libsctp-dev
```

### Issue: Port conflicts

```bash
# Check what's using a port
sudo netstat -tulpn | grep :PORT_NUMBER

# Kill process if needed
sudo kill -9 PID
```

### Issue: Network connectivity

```bash
# Check network interfaces
ip addr show

# Test connectivity
ping 8.8.8.8

# Check firewall rules
sudo ufw status
```

## Next Steps

1. **Complete the Tutorial Series**:
   - Follow [Development Guide](DEVELOPMENT_GUIDE.md)
   - Read component-specific documentation
   - Try example implementations

2. **Join the Community**:
   - Participate in GitHub Discussions
   - Report issues and suggestions
   - Contribute to the project

3. **Experiment and Learn**:
   - Modify configurations
   - Implement new features
   - Run performance tests
   - Document your findings

4. **Advanced Topics**:
   - Network slicing implementation
   - RIC and xApp development
   - AI/ML integration
   - 6G feature prototyping

## Additional Resources

### Documentation
- [Development Guide](DEVELOPMENT_GUIDE.md)
- [Testing Guide](TESTING_GUIDE.md)
- [API Reference](../references/API_REFERENCE.md)

### External Resources
- [srsRAN Documentation](https://docs.srsran.com/)
- [Open5GS Quickstart](https://open5gs.org/open5gs/docs/guide/01-quickstart/)
- [O-RAN Software Community](https://wiki.o-ran-sc.org/)
- [3GPP Specifications](https://www.3gpp.org/specifications)

### Video Tutorials
- YouTube: Search for "5G RAN tutorial"
- YouTube: Search for "O-RAN architecture"
- YouTube: Search for "Open5GS deployment"

## Getting Help

If you encounter issues:

1. **Check the Documentation**: Most common questions are answered in the docs
2. **Search Issues**: Look for similar issues on GitHub
3. **Ask for Help**: Create a new issue with detailed information
4. **Community Discussion**: Join discussions on GitHub

## Summary Checklist

- [ ] System updated and prepared
- [ ] Docker installed and tested
- [ ] Kubernetes (minikube) running
- [ ] Python environment set up
- [ ] Repository cloned
- [ ] srsRAN built and installed
- [ ] Open5GS installed
- [ ] IDE configured
- [ ] Documentation reviewed
- [ ] First test completed

Congratulations! You're now ready to start working with 6G OPENRAN. Happy learning!
