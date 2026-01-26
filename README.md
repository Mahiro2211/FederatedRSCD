# 🌍 Remote Sensing Change Detection withFederated Learning


</br>

</hr>


# Introducion

[English](readme_doc/readme_EN.md) | [简体中文](readme_doc/readme_CN.md)

# Requirement
* Python: 3.10

```python
pip install -r requirement.txt
```

![stars](https://img.shields.io/github/stars/Mahiro2211/FederatedRSCD)
# Dataset
```
Datasets
├── LEVIR
│   ├── Total Clients: 2
│   ├── Client 1
│   │   ├── Dataset: LEVIR
│   │   ├── Training Samples: 2,563
│   │   ├── Sampler: Random
│   └── Client 2
│       ├── Dataset: LEVIR
│       ├── Training Samples: 1,139
│       ├── Sampler: Weighted
│       └── Weights: Default
│
├── S2Looking
│   ├── Total Clients: 4
│   ├── Client 3
│   │   ├── Dataset: S2Looking
│   │   ├── Training Samples: 14,000
│   │   └── Sampler: Random
│   ├── Client 4
│   │   ├── Dataset: S2Looking
│   │   ├── Training Samples: 5,040
│   │   └── Sampler: Sequential
│   ├── Client 5
│   │   ├── Dataset: S2Looking
│   │   ├── Training Samples: 1,260
│   │   └── Sampler: Random
│   └── Client 6
│       ├── Dataset: S2Looking
│       ├── Training Samples: 140
│       └── Sampler: Weighted
│
└── WHUCD
    ├── Total Clients: 2
    ├── Client 7
    │   ├── Dataset: WHUCD
    │   ├── Training Samples: 1,245
    │   └── Sampler: Random
    └── Client 8
        ├── Dataset: WHUCD
        ├── Training Samples: 1,245
        └── Sampler: Sequential
```

<hr>

```
Summary
├── Total Datasets: 3
├── Total Clients: 8
└── Total Training Samples: 26,632
```
# About
This client partitioning introduces both data volume imbalance and sampler heterogeneity, forming a realistic Non-IID federated learning benchmark.

![alt text](elements/image.png)

## 🤖 AI-assisted Development

This project was developed with the assistance of AI tools (GLM 4.7) for:
- Code structuring and refactoring
- Documentation drafting and polishing
- Debugging and design discussions

All model design, experiments, and final decisions were made by the author.
