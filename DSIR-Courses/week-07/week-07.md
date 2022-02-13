- **Required Preparation** will include installing new packages, downloading data, and so on. This is required.
- **Optional Prework** will contain videos, reading, and other resources that may help prepare you for the following week's content. This is optional, but it is recommended that you review the materials shared here!

## Required Preparation:
- You need to install `tensorflow` and `keras`. To make sure this worked:
	- Run `pip install tensorflow --upgrade`.
	- Run `pip install keras --upgrade`.
	- In a fresh Jupyter Notebook, run the following snippet of code:

```python
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import plot_model

X, y = make_regression(n_samples=10000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

model = Sequential()

model.add(Dense(32, # How many neurons do you want in your first layer?
                input_shape=(20,),
                activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(X_train_sc, y_train, validation_data=(X_test_sc, y_test), epochs=10, batch_size=512)
```

## Additional Installs
To visualize your networks you'll need to install the following libraries
Pydot
- ```pip install pydot```

Graphviz
- Follow the instructions [here](https://graphviz.gitlab.io/download/)

After running the starter code run each of the below in a separate cell
- ```model.summary()```
- ```plot_model(model, show_layer_names=True, show_shapes=True)```

## Optional Prework:

- [3Blue1Brown video series on neural networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) is a great introduction to neural networks.
- [This link](http://neuralnetworksanddeeplearning.com/chap1.html) is a good primer on neural networks.
