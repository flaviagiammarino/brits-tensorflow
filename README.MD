# BRITS TensorFlow

![license](https://img.shields.io/github/license/flaviagiammarino/brits-tensorflow)
![languages](https://img.shields.io/github/languages/top/flaviagiammarino/brits-tensorflow)

TensorFlow implementation of multivariate time series imputation model introduced in Cao, W., Wang, D., Li, J.,
Zhou, H., Li, L. and Li, Y., 2018. BRITS: Bidirectional recurrent imputation for time series.
*Advances in neural information processing systems*, 31. 
[NeurIPS:734e6bfcd358e25ac1db0a4241b95651](https://proceedings.neurips.cc/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf)

## Dependencies
```bash
pandas==1.4.3
numpy==1.23.1
tensorflow==2.9.1
plotly==5.9.0
kaleido==0.2.1
```
## Usage
```python
import numpy as np
from brits_tensorflow.model import BRITS

# Generate two time series
N = 2000
t = np.linspace(0, 1, N)
e = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=N)
a = 40 + 20 * np.cos(2 * np.pi * (10 * t - 0.5)) + e[:, 0]
b = 60 + 30 * np.cos(2 * np.pi * (20 * t - 0.5)) + e[:, 1]
x = np.hstack([a.reshape(- 1, 1), b.reshape(- 1, 1)])

# Add some missing values
x[np.random.randint(low=0, high=N, size=int(0.3 * N)), 0] = np.nan
x[np.random.randint(low=0, high=N, size=int(0.3 * N)), 1] = np.nan

# Fit the model
model = BRITS(
    x=x,
    units=100,
    timesteps=200
)

model.fit(
    learning_rate=0.001,
    batch_size=16,
    epochs=200,
    verbose=True
)

# Impute the missing values
imputations = model.predict(x=x)

# Plot the imputations
fig = model.plot_imputations()
fig.write_image('imputations.png', width=1000, height=650)
```
![imputations](example/imputations.png)