import pickle
import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_labels(label_file):
    with open(label_file, 'rb') as f:
        labels = pickle.load(f)
    print("Loaded labels:", labels)  # Pour vérifier la structure des labels
    return labels.tolist()  # Convertir en liste si c'est un tableau numpy



def create_generators():
    labels = load_labels(config.LABELS_FILE)
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(150, 150),
        batch_size=config.BATCH_SIZE,
        classes=labels,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        config.VALID_DIR,
        target_size=(150, 150),
        batch_size=config.BATCH_SIZE,
        classes=labels,
        class_mode='categorical'
    )

    return train_generator, valid_generator
