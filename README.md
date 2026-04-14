# self-driving-car-simulation

## Getting Started

### 1. Clone the Repository

```bash
git clone git@github.com:zoro-sishui/self-driving-car-simulation.git
```

### 2. Setup Environment

#### 1. Linux/Mac

```bash
cd self-driving-car-simulation
python3.10 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

#### 2. Windows

```bash
cd self-driving-car-simulation
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

### 3. Data Download

Download here: [Download Dataset](https://drive.google.com/file/d/1IQBwh6KJnm8Xo2kL9ozBaYWfLU59rUn-/view?usp=drive_link)

Place the extracted dataset files in `data/` so `data/driving_log.csv` and `IMG` directory is available.

### 4. Run Scripts

#### 1. Check Dataset Distribution

`histogram.py` can be run independently to inspect steering-angle distribution and produce a balanced CSV.

```bash
python scripts/histogram.py
```

Outputs:

- `data/balanced_driving_log.csv`

- `outputs/steering_histogram.png`

#### 2. Train the Model

```bash
python scripts/train.py
```

Outputs:

- `models/model.h5`

- `loss.png`

#### 3. Test the Model

```bash
python scripts/test.py
```

This starts a Socket.IO server on port `4567` and loads `models/model.h5` for inference.

### 5. Commit Message Guidelines

Use clear commit messages in this format:

- feat: add new feature
- fix: fix bug
- docs: update documentation
- refactor: improve code
