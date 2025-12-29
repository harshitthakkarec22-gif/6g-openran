# 5G/6G Core Network

This directory contains the implementation of the 5G/6G Core Network (5GC/6GC) based on Service-Based Architecture (SBA).

## Overview

The core network follows 3GPP specifications with service-based architecture:

```
┌──────────────────────────────────────────────────────────────┐
│                    Service Bus (SBI)                         │
│                  HTTP/2 RESTful APIs                         │
└────┬──────┬──────┬──────┬──────┬──────┬──────┬──────────────┘
     │      │      │      │      │      │      │
┌────▼──┐ ┌─▼───┐ ┌▼───┐ ┌▼───┐ ┌▼───┐ ┌▼───┐ ┌▼────┐
│  AMF  │ │ SMF │ │PCF │ │UDM │ │AUSF│ │NRF │ │ NEF │
│Access │ │Sess │ │Pol │ │Data│ │Auth│ │Disc│ │Expo │
└───┬───┘ └──┬──┘ └────┘ └────┘ └────┘ └────┘ └─────┘
    │        │
    │N2      │N4
    ▼        ▼
┌────────┐ ┌────────┐
│  gNB   │ │  UPF   │──N6──► Internet/DN
│  RAN   │ │  User  │
└────────┘ │  Plane │
           └────────┘
```

## Network Functions

### Control Plane Functions

#### AMF (Access and Mobility Management Function)

Location: `core/amf/`

**Responsibilities**:
- Registration management
- Connection management
- Reachability management
- Mobility management
- Access authentication and authorization

**Key Interfaces**:
- N1: UE ↔ AMF (NAS)
- N2: RAN ↔ AMF (NGAP)
- Namf: Service-based interface

**Configuration**:
```yaml
# core/amf/config/amf.yaml
amf:
  plmn_id:
    mcc: "001"
    mnc: "01"
  region_id: 128
  set_id: 1
  ngap:
    bind_addr: "0.0.0.0"
    port: 38412
  sbi:
    bind_addr: "0.0.0.0"
    port: 80
    scheme: "http"
```

#### SMF (Session Management Function)

Location: `core/smf/`

**Responsibilities**:
- PDU session establishment, modification, release
- UE IP address allocation
- DHCP functions
- Selection and control of UPF
- Charging data collection

**Key Interfaces**:
- N4: SMF ↔ UPF (PFCP)
- N7: SMF ↔ PCF
- Nsmf: Service-based interface

**Configuration**:
```yaml
# core/smf/config/smf.yaml
smf:
  sbi:
    bind_addr: "0.0.0.0"
    port: 80
  pfcp:
    bind_addr: "0.0.0.0"
    port: 8805
  subnet:
    - cidr: "10.45.0.0/16"
      dnn: "internet"
  dns:
    - "8.8.8.8"
    - "8.8.4.4"
```

#### PCF (Policy Control Function)

Location: `core/pcf/`

**Responsibilities**:
- Policy rule decisions
- QoS control
- Charging control
- Network slice selection assistance

**Key Interfaces**:
- N5: PCF ↔ AF (Application Function)
- N7: PCF ↔ SMF
- Npcf: Service-based interface

#### UDM (Unified Data Management)

Location: `core/udm/`

**Responsibilities**:
- User subscription data management
- Authentication credential processing
- User consent management
- SMS management

**Key Interfaces**:
- Nudm: Service-based interface

#### AUSF (Authentication Server Function)

Location: `core/ausf/`

**Responsibilities**:
- Authentication services
- Key derivation
- Security context management

**Key Interfaces**:
- Nausf: Service-based interface

#### NRF (Network Repository Function)

Location: `core/nrf/`

**Responsibilities**:
- Service registration
- Service discovery
- Network function profile management

**Key Interfaces**:
- Nnrf: Service-based interface

**Configuration**:
```yaml
# core/nrf/config/nrf.yaml
nrf:
  sbi:
    bind_addr: "0.0.0.0"
    port: 80
  mongodb:
    uri: "mongodb://localhost:27017"
    database: "open5gs"
```

### User Plane Function

#### UPF (User Plane Function)

Location: `core/upf/`

**Responsibilities**:
- Packet routing and forwarding
- Traffic steering and routing
- QoS handling
- Buffering
- Downlink packet notification
- Usage reporting

**Key Interfaces**:
- N3: RAN ↔ UPF (GTP-U)
- N4: SMF ↔ UPF (PFCP)
- N6: UPF ↔ Data Network
- N9: UPF ↔ UPF (for mobility)

**Configuration**:
```yaml
# core/upf/config/upf.yaml
upf:
  pfcp:
    bind_addr: "0.0.0.0"
    port: 8805
  gtpu:
    bind_addr: "0.0.0.0"
  subnet:
    - cidr: "10.45.0.0/16"
      dnn: "internet"
  metrics:
    port: 9090
```

## Service-Based Architecture (SBA)

### Service Registration

```python
# Example: NF registering with NRF
def register_nf():
    profile = {
        "nfInstanceId": "amf-001",
        "nfType": "AMF",
        "nfStatus": "REGISTERED",
        "plmnList": [{
            "mcc": "001",
            "mnc": "01"
        }],
        "sNssais": [{
            "sst": 1,
            "sd": "000001"
        }],
        "ipv4Addresses": ["192.168.1.10"],
        "nfServices": [{
            "serviceInstanceId": "namf-comm",
            "serviceName": "namf-comm",
            "versions": [{
                "apiVersionInUri": "v1",
                "apiFullVersion": "1.0.0"
            }],
            "scheme": "http",
            "nfServiceStatus": "REGISTERED"
        }]
    }
    
    response = requests.put(
        "http://nrf:80/nnrf-nfm/v1/nf-instances/amf-001",
        json=profile
    )
    return response.json()
```

### Service Discovery

```python
# Example: Discovering SMF instances
def discover_smf():
    params = {
        "target-nf-type": "SMF",
        "requester-nf-type": "AMF",
        "limit": 10
    }
    
    response = requests.get(
        "http://nrf:80/nnrf-disc/v1/nf-instances",
        params=params
    )
    return response.json()
```

## Call Flows

### Registration Procedure

```
UE        RAN       AMF       AUSF      UDM       SMF
│          │         │         │         │         │
├─Registration──────►│         │         │         │
│  Request  │        │         │         │         │
│          │         │         │         │         │
│          ├────N2───►│         │         │         │
│          │         │         │         │         │
│          │         ├─Nausf───►│         │         │
│          │         │ Auth Req │         │         │
│          │         │         │         │         │
│          │         │         ├─Nudm───►│         │
│          │         │         │ Get Auth│         │
│          │         │         │         │         │
│          │         │         │◄────────┤         │
│          │         │         │Auth Data│         │
│          │         │         │         │         │
│          │         │◄────────┤         │         │
│          │         │Auth Resp│         │         │
│          │         │         │         │         │
│◄────────────────────┤         │         │         │
│  Auth Request       │         │         │         │
│          │         │         │         │         │
├─────────────────────►│         │         │         │
│  Auth Response      │         │         │         │
│          │         │         │         │         │
│          │         ├─Nudm─────────────►│         │
│          │         │ Register │         │         │
│          │         │         │         │         │
│◄────────────────────┤         │         │         │
│  Registration Accept│         │         │         │
```

### PDU Session Establishment

```
UE        RAN       AMF       SMF       UPF       PCF
│          │         │         │         │         │
├─PDU Session────────►│         │         │         │
│  Establishment      │         │         │         │
│          │         │         │         │         │
│          │         ├─Nsmf───►│         │         │
│          │         │Create   │         │         │
│          │         │         │         │         │
│          │         │         ├─Npcf───►│         │
│          │         │         │Policy   │         │
│          │         │         │         │         │
│          │         │         ├─N4─────►│         │
│          │         │         │Session  │         │
│          │         │         │Establish│         │
│          │         │         │         │         │
│          │         │◄────────┤         │         │
│          │         │Response │         │         │
│          │         │         │         │         │
│          ├────N2───┤         │         │         │
│          │Session  │         │         │         │
│          │Setup    │         │         │         │
│          │         │         │         │         │
│◄─────────┤         │         │         │         │
│ Session  │         │         │         │         │
│ Accepted │         │         │         │         │
│          │         │         │         │         │
├──────Data Traffic──────────────────────►│         │
│          │         │         │         │Data Fwd │
```

## Deployment

### Docker Compose Deployment

```yaml
# core/docker-compose.yml
version: '3.8'

services:
  mongodb:
    image: mongo:5.0
    volumes:
      - db_data:/data/db
  
  nrf:
    build: ./nrf
    depends_on:
      - mongodb
    ports:
      - "7777:80"
  
  amf:
    build: ./amf
    depends_on:
      - nrf
    ports:
      - "38412:38412/sctp"
  
  smf:
    build: ./smf
    depends_on:
      - nrf
  
  upf:
    build: ./upf
    cap_add:
      - NET_ADMIN
    devices:
      - /dev/net/tun
  
  ausf:
    build: ./ausf
    depends_on:
      - nrf
  
  udm:
    build: ./udm
    depends_on:
      - nrf
  
  pcf:
    build: ./pcf
    depends_on:
      - nrf

volumes:
  db_data:
```

### Kubernetes Deployment

```bash
# Deploy core network
kubectl apply -f core/k8s/

# Check deployment
kubectl get pods -n core-network
kubectl get services -n core-network

# View logs
kubectl logs -f deployment/amf -n core-network
```

## Network Slicing

### Slice Configuration

```yaml
# Example slice configuration
slices:
  - sst: 1  # eMBB
    sd: "000001"
    description: "Enhanced Mobile Broadband"
    qos:
      5qi: 9
      arp: 8
    resources:
      guaranteed_bitrate: "100Mbps"
      max_bitrate: "1Gbps"
  
  - sst: 2  # URLLC
    sd: "000002"
    description: "Ultra-Reliable Low-Latency"
    qos:
      5qi: 82
      arp: 1
    resources:
      latency: "1ms"
      reliability: "99.9999%"
  
  - sst: 3  # mMTC
    sd: "000003"
    description: "Massive IoT"
    qos:
      5qi: 6
      arp: 15
    resources:
      connection_density: "1000000/km2"
```

## Testing

### Unit Tests

```bash
# Test AMF
cd core/amf/tests
pytest test_amf.py

# Test SMF
cd core/smf/tests
pytest test_smf.py

# Test UPF
cd core/upf/tests
pytest test_upf.py
```

### Integration Tests

```bash
# Start core network
docker-compose -f core/docker-compose.yml up -d

# Run integration tests
cd testing/integration-tests
pytest test_core_integration.py

# Stop core network
docker-compose -f core/docker-compose.yml down
```

## Monitoring

### Metrics

```bash
# Prometheus metrics
curl http://localhost:9090/metrics

# Grafana dashboard
http://localhost:3000
```

### Key Metrics to Monitor

- **AMF**: Registered UEs, active connections
- **SMF**: Active PDU sessions, throughput
- **UPF**: Packet forwarding rate, latency
- **NRF**: Registered NFs, discovery requests

## Troubleshooting

### Common Issues

**Issue: NF registration failed**
```bash
# Check NRF connectivity
curl http://nrf:80/nnrf-nfm/v1/nf-instances

# Verify network
docker network inspect core-network

# Check logs
docker logs nrf
```

**Issue: No PDU session established**
```bash
# Check SMF-UPF PFCP association
tcpdump -i any port 8805

# Verify routing
ip route

# Check UPF logs
docker logs upf
```

## References

- [3GPP TS 23.501: 5G System Architecture](https://www.3gpp.org/ftp/Specs/archive/23_series/23.501/)
- [3GPP TS 23.502: 5G Procedures](https://www.3gpp.org/ftp/Specs/archive/23_series/23.502/)
- [3GPP TS 29.500: 5G Service-Based Architecture](https://www.3gpp.org/ftp/Specs/archive/29_series/29.500/)
- [Open5GS Documentation](https://open5gs.org/)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.

## License

See [LICENSE](../../LICENSE) file in the root directory.
