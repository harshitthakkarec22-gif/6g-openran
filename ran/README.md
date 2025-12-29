# Radio Access Network (RAN)

This directory contains the implementation of the Radio Access Network components following O-RAN disaggregation principles.

## Overview

The RAN is disaggregated into three main functional units:

```
┌─────────────────────────────────────────────┐
│         Centralized Unit (CU)               │
│  ┌──────────────┐    ┌──────────────┐      │
│  │   CU-CP      │    │   CU-UP      │      │
│  │  (Control)   │◄───┤   (User)     │      │
│  └──────┬───────┘ E1 └──────┬───────┘      │
└─────────┼──────────────────┼────────────────┘
          │ F1-C             │ F1-U
          ▼                  ▼
┌─────────────────────────────────────────────┐
│       Distributed Unit (DU)                 │
│  • MAC (Scheduler)                          │
│  • RLC (Segmentation)                       │
│  • High PHY                                 │
└─────────┬───────────────────────────────────┘
          │ Open Fronthaul (eCPRI)
          ▼
┌─────────────────────────────────────────────┐
│         Radio Unit (RU)                     │
│  • Low PHY (FFT/IFFT)                       │
│  • RF Processing                            │
│  • Beamforming                              │
└─────────────────────────────────────────────┘
```

## Components

### Centralized Unit (CU)

Location: `ran/cu/`

The CU handles non-real-time L2 and L3 protocols:

- **CU-CP (Control Plane)**:
  - RRC (Radio Resource Control)
  - PDCP-C (Control plane PDCP)
  - Interfaces: F1-C, N2, Xn

- **CU-UP (User Plane)**:
  - SDAP (Service Data Adaptation Protocol)
  - PDCP-U (User plane PDCP)
  - Interfaces: F1-U, N3, E1

**Key Functions**:
- Connection management
- Mobility control
- Security (ciphering, integrity)
- QoS flow mapping

### Distributed Unit (DU)

Location: `ran/du/`

The DU handles real-time L1/L2 protocols:

- **MAC Layer**:
  - Resource scheduling
  - HARQ management
  - Logical channel prioritization
  - Random access

- **RLC Layer**:
  - Segmentation/reassembly
  - ARQ (Automatic Repeat Request)
  - In-sequence delivery

- **High PHY**:
  - Channel coding (LDPC/Polar)
  - Rate matching
  - Scrambling
  - Modulation

**Key Functions**:
- Real-time scheduling (<1ms)
- Resource block allocation
- Link adaptation
- Power control

### Radio Unit (RU)

Location: `ran/ru/`

The RU handles physical layer RF processing:

- **Low PHY**:
  - FFT/IFFT operations
  - CP insertion/removal
  - Precoding
  - Resource element mapping

- **RF Frontend**:
  - Digital-to-analog conversion
  - Power amplification
  - Filtering
  - Frequency conversion

- **Antenna System**:
  - Massive MIMO
  - Digital beamforming
  - Beam management

**Key Functions**:
- RF signal generation
- Beamforming
- Antenna calibration
- Interference management

## Interfaces

### F1 Interface (CU ↔ DU)

**F1-C (Control Plane)**:
- Protocol: SCTP
- Messages: F1AP
- Functions:
  - UE context management
  - RRC message transfer
  - Paging
  - System information

**F1-U (User Plane)**:
- Protocol: GTP-U/UDP
- Functions:
  - Data forwarding
  - Flow control
  - QoS enforcement

### Open Fronthaul Interface (DU ↔ RU)

**Functional Split**: 7.2x (intra-PHY split)

**Protocols**:
- C-Plane: Control and synchronization
- U-Plane: User data (IQ samples)
- S-Plane: Synchronization
- M-Plane: Management (NETCONF/YANG)

**Transport**: eCPRI over Ethernet (10G/25G/100G)

### E1 Interface (CU-CP ↔ CU-UP)

**Purpose**: Separate control and user plane in CU

**Protocol**: SCTP/E1AP

**Functions**:
- Bearer context management
- QoS flow management
- Data forwarding control

## Implementation Guide

### Prerequisites

```bash
# Install dependencies
sudo apt-get install -y \
    libfftw3-dev \
    libmbedtls-dev \
    libsctp-dev \
    libyaml-cpp-dev

# For simulation
pip install numpy scipy matplotlib
```

### Building CU

```bash
cd ran/cu
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Building DU

```bash
cd ran/du
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Building RU

```bash
cd ran/ru
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Configuration

### CU Configuration Example

```yaml
# ran/cu/config/cu.yaml
cu:
  id: 1
  name: "CU-001"
  
  cu_cp:
    plmn_id: "00101"
    tac: 1
    cell_list:
      - cell_id: 1
        pci: 1
        dl_earfcn: 3350
    
  cu_up:
    gtpu_endpoint: "127.0.0.1"
    gtpu_port: 2152
    
  interfaces:
    f1c_bind: "127.0.0.1:38472"
    n2_amf: "127.0.0.2:38412"
```

### DU Configuration Example

```yaml
# ran/du/config/du.yaml
du:
  id: 1
  name: "DU-001"
  
  mac:
    scheduler: "round-robin"
    harq_processes: 8
    
  rlc:
    mode: "acknowledged"
    sn_length: 12
    
  interfaces:
    f1c_cu: "127.0.0.1:38472"
    fronthaul_ru: "127.0.0.1:50000"
```

### RU Configuration Example

```yaml
# ran/ru/config/ru.yaml
ru:
  id: 1
  name: "RU-001"
  
  rf:
    tx_gain: 80
    rx_gain: 40
    frequency: 3500000000  # 3.5 GHz
    
  antenna:
    num_tx: 64
    num_rx: 64
    beamforming: true
    
  fronthaul:
    du_endpoint: "127.0.0.1:50000"
    vlan_id: 100
```

## Testing

### Unit Tests

```bash
# Test CU
cd ran/cu/tests
pytest test_cu.py

# Test DU
cd ran/du/tests
pytest test_du.py

# Test RU
cd ran/ru/tests
pytest test_ru.py
```

### Integration Tests

```bash
# Start all components
./scripts/start-ran.sh

# Run integration tests
cd testing/integration-tests
pytest test_ran_integration.py

# Stop components
./scripts/stop-ran.sh
```

## Performance Metrics

### Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| **F1 Latency** | < 10ms | CU-DU communication |
| **Fronthaul Latency** | < 100μs | DU-RU communication |
| **Scheduling Latency** | < 1ms | MAC scheduler |
| **Throughput** | 1+ Gbps | Per UE |
| **RU Processing** | < 50μs | PHY layer |

### Monitoring

```bash
# Monitor RAN metrics
kubectl top pods -l component=ran

# View logs
kubectl logs -f ran-cu-0
kubectl logs -f ran-du-0
kubectl logs -f ran-ru-0

# Prometheus metrics
curl http://localhost:9090/metrics
```

## Development

### Adding New Features

1. **CU Features**:
   - Modify `ran/cu/src/`
   - Update configuration schema
   - Add unit tests
   - Update documentation

2. **DU Features**:
   - Modify scheduler: `ran/du/src/mac/scheduler.cpp`
   - Update RLC: `ran/du/src/rlc/`
   - Add tests

3. **RU Features**:
   - Update PHY: `ran/ru/src/phy/`
   - Modify RF: `ran/ru/src/rf/`
   - Test with simulator

### Code Structure

```
ran/
├── cu/
│   ├── src/
│   │   ├── cu_cp/       # Control plane
│   │   ├── cu_up/       # User plane
│   │   ├── rrc/         # RRC protocol
│   │   └── pdcp/        # PDCP layer
│   ├── include/         # Header files
│   ├── config/          # Configuration files
│   └── tests/           # Unit tests
├── du/
│   ├── src/
│   │   ├── mac/         # MAC layer
│   │   ├── rlc/         # RLC layer
│   │   └── phy/         # High PHY
│   ├── include/
│   ├── config/
│   └── tests/
└── ru/
    ├── src/
    │   ├── phy/         # Low PHY
    │   ├── rf/          # RF processing
    │   └── beam/        # Beamforming
    ├── include/
    ├── config/
    └── tests/
```

## Troubleshooting

### Common Issues

**Issue: F1 connection failed**
```bash
# Check network connectivity
ping <CU_IP>

# Verify ports are open
netstat -tulpn | grep 38472

# Check firewall
sudo ufw status
```

**Issue: High latency on fronthaul**
```bash
# Check network interface
ethtool <interface>

# Verify VLAN configuration
ip link show

# Test with iperf
iperf3 -c <RU_IP>
```

**Issue: Low throughput**
```bash
# Check CPU usage
top -H

# Verify scheduler configuration
# Review logs for retransmissions
# Check RF parameters
```

## References

- [O-RAN WG4: Open Fronthaul Specification](https://www.o-ran.org/)
- [3GPP TS 38.470: F1 Interface](https://www.3gpp.org/ftp/Specs/archive/38_series/38.470/)
- [3GPP TS 38.300: NR Overall Description](https://www.3gpp.org/ftp/Specs/archive/38_series/38.300/)
- [srsRAN Documentation](https://docs.srsran.com/)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines on:
- Code style
- Pull request process
- Testing requirements
- Documentation standards

## License

See [LICENSE](../../LICENSE) file in the root directory.
