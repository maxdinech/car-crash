
# Modèle 1 : CNN à une convolution. ~98.8% après 12 epochs

model = Sequential([
    Convolution2D(32, (5,5), input_shape=(40,40,1), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(43, activation='softmax')
])

# Modèle 2 : CNN à deux convolutions. 99.5% après 15 epochs

model = Sequential([
    Convolution2D(20, (5,5), input_shape=(40,40,1), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Convolution2D(40, (5,5), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(100, activation='relu'),
    Dropout(0.4),
    Dense(43, activation='softmax')
])
