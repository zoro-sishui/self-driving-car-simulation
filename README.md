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

---

## Challenges & Solutions

### Car steering toward the edge on turns

**Problem:**
During initial testing in the simulator, the car would drive normally on straight sections but would drive toward the edge of the road on turn. The model was not correcting its steering angle enough when curves appeared.

**Root cause:**
The training dataset was skewed toward straight driving, leaving the model with less exposure to turning scenarios. As a result it defaulted to minimal steering corrections and improperly drove around the curves.

**How we fixed it:**
We plotted a histogram of the steering angle distribution (`scripts/histogram.py`), which made the imbalance immediately visible. We applied undersampling by capping the number of samples per bin, producing a balanced dataset saved to `data/balanced_driving_log.csv`.

Additionally, the `pan` augmentation in `scripts/augmentation.py` artificially shifts images left or right to simulate the car being off-centre. Since shifting the image changes where the car appears to be on the road, the steering label is updated to match: `steering += (dx / max_shift_x) * 0.4`. This teaches the model that being off-centre requires a corrective input, improving its ability to recover during turns rather than continuing straight off the road.

---

### 5. Commit Message Guidelines

Use clear commit messages in this format:

- feat: add new feature
- fix: fix bug
- docs: update documentation
- refactor: improve code
