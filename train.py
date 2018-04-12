import Models
import LoadBatches

train_images_path = 'data/train/'
train_segs_path = 'data/label/'
train_batch_size = 1
n_classes = 2
input_height = 96
input_width = 64
validate = False
save_weights_path = 'results/'
epochs = 100
optimizer_name = 'adam'

val_images_path = ''
val_segs_path = ''
val_batch_size = 2

model = Models.Unet(n_classes, input_height=input_height, input_width=input_width)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_name,
              metrics=['accuracy'])

print("Model output shape", model.output_shape)
model.summary()
output_height = 96
output_width = 64

G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                           input_height, input_width, output_height, output_width)

if validate:
    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height,
                                                input_width, output_height, output_width)

if not validate:
    for ep in range(epochs):
        print('current epoch:', ep)
        model.fit_generator(G, 400//train_batch_size, epochs=1)
        model.save_weights(save_weights_path + "model_" + str(ep) + ".h5")
        model.save(save_weights_path + "model_" + str(ep) + ".json")
else:
    for ep in range(epochs):
        model.fit_generator(G, 512, validation_data=G2, validation_steps=200, epochs=1)
        model.save_weights(save_weights_path + "." + str(ep))
        model.save(save_weights_path + "model_" + str(ep) + ".h5")
