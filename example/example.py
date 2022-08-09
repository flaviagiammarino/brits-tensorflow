import numpy as np
from brits_tensorflow.model import BRITS

# Generate two time series
N = 1000
t = np.linspace(0, 1, N)
e = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=N)
a = 40 + 20 * np.cos(2 * np.pi * (10 * t - 0.5)) + e[:, 0]
b = 60 + 30 * np.cos(2 * np.pi * (20 * t - 0.5)) + e[:, 1]
x = np.hstack([a.reshape(- 1, 1), b.reshape(- 1, 1)])

# Add some missing values
x[np.random.randint(low=0, high=N, size=int(0.2 * N)), 0] = np.nan
x[np.random.randint(low=0, high=N, size=int(0.2 * N)), 1] = np.nan

# Fit the model
model = BRITS(
    x=x,
    units=100,
    timesteps=100
)

model.fit(
    learning_rate=0.0001,
    batch_size=4,
    epochs=100,
    verbose=True
)

# Plot the imputations
imputations = model.predict(x=x)
fig = model.plot_imputations()
fig.write_image('imputations.png', width=1000, height=650)
