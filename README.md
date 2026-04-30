# Soft Sensors - GeoDSTG

Official implementation of GeoDSTG: Geometric Dynamic Spatio-Temporal Graph Neural Network for Soft Sensor Modeling.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```python
from main import Config, setup_environment
from models.GeoDSTG import GeoDSTG

# Setup
device = setup_environment()
