from data_loader import load_labels
import config

def predict(img_path):
    model = load_model('plant_disease_model.h5')
    labels = load_labels(config.LABELS_FILE)  # Charger les labels pour l'interprétation

    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = labels[predicted_class_index]
    return predicted_class

if __name__ == "__main__":
    img_path = '/test/test/AppleCedarRust1.JPG'
    print(predict(img_path))
