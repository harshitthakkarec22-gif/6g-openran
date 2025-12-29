# 6G OPENRAN Architecture

A comprehensive boilerplate repository for 6G Open Radio Access Network (OPENRAN) architecture, designed for undergraduate students and researchers.

## 📚 Overview

This repository provides a complete framework for understanding, learning, and implementing 6G OPENRAN systems. It includes:

- **Comprehensive Documentation**: Detailed research report with methodology and tools
- **Modular Architecture**: Well-organized component structure following O-RAN specifications
- **Learning Resources**: References, guides, and examples for students
- **Practical Implementation**: Boilerplate code and configuration templates

## 🎯 Key Features

- ✅ **O-RAN Compliant**: Follows O-RAN Alliance specifications
- ✅ **Disaggregated RAN**: CU/DU/RU separation
- ✅ **5G Core Network**: Complete 5GC implementation structure
- ✅ **RAN Intelligent Controller (RIC)**: AI/ML-based network optimization
- ✅ **Network Slicing**: Support for eMBB, URLLC, and mMTC
- ✅ **6G Features**: THz communication, AI-native architecture, Digital Twin
- ✅ **Cloud-Native**: Containerized, microservices-based design
- ✅ **Undergraduate-Friendly**: Comprehensive learning materials

## 📖 Documentation

### Essential Reading

1. **[Research Report](docs/RESEARCH_REPORT.md)** - Comprehensive guide covering:
   - Introduction to 6G OPENRAN
   - Research methodology
   - Tools and technologies
   - Implementation roadmap
   - Learning resources

2. **[Initial Architecture](docs/architecture/INITIAL_ARCHITECTURE.md)** - Architecture design:
   - System architecture diagrams
   - Component specifications
   - Interface definitions
   - Deployment models

### Quick Links

- [Getting Started Guide](docs/guides/GETTING_STARTED.md)
- [Development Guide](docs/guides/DEVELOPMENT_GUIDE.md)
- [Testing Guide](docs/guides/TESTING_GUIDE.md)
- [References](docs/references/REFERENCES.md)

## 🏗️ Repository Structure

```
6g-openran/
├── ran/                        # Radio Access Network components
│   ├── cu/                     # Centralized Unit (CU-CP, CU-UP)
│   ├── du/                     # Distributed Unit (MAC, RLC, PHY)
│   └── ru/                     # Radio Unit (RF, antenna)
├── core/                       # 5G Core Network functions
│   ├── amf/                    # Access and Mobility Management
│   ├── smf/                    # Session Management
│   ├── upf/                    # User Plane Function
│   ├── ausf/                   # Authentication Server
│   ├── udm/                    # Unified Data Management
│   ├── pcf/                    # Policy Control
│   └── nrf/                    # Network Repository Function
├── management/                 # Management & Orchestration
│   ├── orchestrator/           # Service orchestration (MANO)
│   ├── monitoring/             # Telemetry and monitoring
│   └── logs/                   # Centralized logging
├── network/                    # Network services
│   ├── slicing/                # Network slicing management
│   └── edge-computing/         # Multi-access Edge Computing (MEC)
├── testing/                    # Testing framework
│   ├── unit-tests/             # Unit tests
│   ├── integration-tests/      # Integration tests
│   └── performance-tests/      # Performance/load tests
├── simulation/                 # Simulation environment
│   ├── scenarios/              # Test scenarios
│   └── traffic-models/         # Traffic generation models
├── docs/                       # Documentation
│   ├── architecture/           # Architecture documents
│   ├── guides/                 # How-to guides
│   └── references/             # Reference materials
├── configs/                    # Configuration files
│   ├── ran/                    # RAN configurations
│   ├── core/                   # Core network configurations
│   └── network/                # Network configurations
├── scripts/                    # Utility scripts
│   ├── setup/                  # Setup scripts
│   ├── deployment/             # Deployment scripts
│   └── utils/                  # Utility scripts
└── examples/                   # Example implementations
    ├── basic-setup/            # Basic setup examples
    └── advanced-scenarios/     # Advanced use cases
```

## 🚀 Quick Start

### Prerequisites

- **Operating System**: Ubuntu 20.04/22.04 LTS (recommended)
- **Hardware**:
  - CPU: 8+ cores
  - RAM: 16GB+ (32GB recommended)
  - Storage: 100GB+ SSD
- **Software**:
  - Docker 20.10+
  - Python 3.8+
  - Git

### Installation

```bash
# Clone the repository
git clone https://github.com/harshitthakkarec22-gif/6g-openran.git
cd 6g-openran

# Run setup script (coming soon)
./scripts/setup/install.sh

# Verify installation
./scripts/utils/check-environment.sh
```

### First Steps

1. **Read the Research Report**: Start with [docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md)
2. **Review the Architecture**: Understand the system design in [docs/architecture/INITIAL_ARCHITECTURE.md](docs/architecture/INITIAL_ARCHITECTURE.md)
3. **Follow Getting Started Guide**: Step-by-step instructions in [docs/guides/GETTING_STARTED.md](docs/guides/GETTING_STARTED.md)
4. **Run Examples**: Try basic examples in the `examples/` directory

## 🛠️ Tools & Technologies

### Core Technologies

| Category | Tools |
|----------|-------|
| **Languages** | Python, C/C++, Go |
| **RAN Software** | srsRAN, OpenAirInterface |
| **Core Network** | Open5GS, free5GC |
| **Simulation** | ns-3, MATLAB |
| **Containerization** | Docker, Kubernetes |
| **Monitoring** | Prometheus, Grafana, ELK |
| **AI/ML** | TensorFlow, PyTorch, scikit-learn |

See the [Research Report](docs/RESEARCH_REPORT.md#5-tools--technologies) for detailed information.

## 📊 Architecture Highlights

### High-Level System View

```
┌─────────────────────────────────────────────────┐
│        Management & Orchestration (SMO)         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Non-RT   │  │ Monitor  │  │ Dashboard│     │
│  │   RIC    │  │          │  │          │     │
│  └────┬─────┘  └──────────┘  └──────────┘     │
└───────┼─────────────────────────────────────────┘
        │ A1
        ▼
┌──────────────────────────────────────────────────┐
│     RAN Intelligent Controller (Near-RT RIC)    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │  xApp   │  │  xApp   │  │  xApp   │         │
│  └────┬────┘  └────┬────┘  └────┬────┘         │
└───────┼────────────┼────────────┼───────────────┘
        │ E2         │            │
        ▼            ▼            ▼
┌──────────────────────────────────────────────────┐
│              Radio Access Network                │
│  ┌─────┐      ┌─────┐      ┌─────┐              │
│  │ CU  │─F1──►│ DU  │─FH──►│ RU  │              │
│  └─────┘      └─────┘      └─────┘              │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│              5G/6G Core Network                  │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐       │
│  │  AMF  │ │  SMF  │ │  UPF  │ │  UDM  │       │
│  └───────┘ └───────┘ └───────┘ └───────┘       │
└──────────────────────────────────────────────────┘
```

See [INITIAL_ARCHITECTURE.md](docs/architecture/INITIAL_ARCHITECTURE.md) for detailed diagrams.

## 🎓 Learning Path

### For Undergraduate Students

1. **Week 1-2**: Fundamentals
   - Read introduction sections
   - Understand basic concepts (RAN, Core, O-RAN)
   - Set up development environment

2. **Week 3-4**: Hands-On Learning
   - Deploy existing open-source tools (srsRAN, Open5GS)
   - Run basic simulations
   - Experiment with configurations

3. **Week 5-8**: Component Development
   - Study individual components (CU, DU, RU, AMF, SMF)
   - Implement simplified versions
   - Test integrations

4. **Week 9-12**: Advanced Topics
   - RIC and xApps
   - Network slicing
   - AI/ML integration
   - 6G-specific features

## 🧪 Testing

```bash
# Run unit tests
./scripts/testing/run-unit-tests.sh

# Run integration tests
./scripts/testing/run-integration-tests.sh

# Run performance tests
./scripts/testing/run-performance-tests.sh
```

## 📈 Roadmap

- [x] Initial repository structure
- [x] Comprehensive research report
- [x] Initial architecture documentation
- [ ] RAN component implementations
- [ ] Core network component implementations
- [ ] RIC framework and xApps
- [ ] Network slicing implementation
- [ ] Simulation environment
- [ ] Testing framework
- [ ] CI/CD pipeline
- [ ] Example applications
- [ ] Video tutorials

## 🤝 Contributing

Contributions are welcome! Please read the [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style guidelines
- Pull request process
- Development workflow
- Testing requirements

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

### Standards & Specifications

- [3GPP Specifications](https://www.3gpp.org/specifications)
- [O-RAN Alliance](https://www.o-ran.org/)
- [ETSI NFV](https://www.etsi.org/technologies/nfv)

### Open Source Projects

- [srsRAN](https://www.srsran.com/) - 4G/5G software radio
- [OpenAirInterface](https://openairinterface.org/) - 5G platform
- [Open5GS](https://open5gs.org/) - 5G core network
- [free5GC](https://free5gc.org/) - 5G core in Go

### Learning Resources

- [O-RAN Software Community](https://wiki.o-ran-sc.org/)
- [3GPP Portal](https://portal.3gpp.org/)
- [IEEE Communications Society](https://www.comsoc.org/)

## 💬 Support

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/harshitthakkarec22-gif/6g-openran/issues)
- **Discussions**: Join discussions in [GitHub Discussions](https://github.com/harshitthakkarec22-gif/6g-openran/discussions)
- **Email**: Contact maintainers for questions

## 🌟 Acknowledgments

This project builds upon the work of:
- O-RAN Alliance and its specifications
- 3GPP standardization efforts
- Open source communities (srsRAN, OpenAirInterface, Open5GS)
- Academic research in 6G technologies

## 📌 Citation

If you use this repository in your research or project, please cite:

```bibtex
@misc{6g-openran-2024,
  title={6G OPENRAN Architecture Boilerplate},
  author={6G OPENRAN Project},
  year={2024},
  publisher={GitHub},
  url={https://github.com/harshitthakkarec22-gif/6g-openran}
}
```

---

**Status**: 🚧 Under Active Development

**Version**: 1.0.0

**Last Updated**: December 2024

For detailed information, please refer to the [Research Report](docs/RESEARCH_REPORT.md) and [Architecture Documentation](docs/architecture/INITIAL_ARCHITECTURE.md).