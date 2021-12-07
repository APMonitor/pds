# Classification neural network with Keras
model = Sequential()
model.add(Dense(8, input_dim=Xtrain.shape[1], activation='relu'))
model.add(Dense(ytrain.shape[1], activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', \
              optimizer='adam', metrics=['accuracy'])

# Train model
result = model.fit(Xtrain,ytrain,epochs=1000,\
                   validation_split=0.2,verbose=0)