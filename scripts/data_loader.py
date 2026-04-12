import os
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'IMG')
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'balanced_driving_log.csv')


def load_data():
    df = pd.read_csv(CSV_PATH, header=None)
    df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

    image_paths = []
    steerings   = []

    for _, row in df.iterrows():
        filename = os.path.basename(row['center'])
        image_paths.append(os.path.normpath(os.path.join(IMG_DIR, filename)))
        steerings.append(float(row['steering']))

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, steerings, test_size=0.2, random_state=42
    )

    print(f'Training samples:   {len(X_train)}')
    print(f'Test samples: {len(X_test)}')

    return X_train, X_test, y_train, y_test