import matplotlib.pyplot as plt
from data_loader import create_generators
from model import build_model
import config

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'ro', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

def train():
    # Création des générateurs
    train_generator, valid_generator = create_generators()

    # Construction du modèle
    model = build_model()

    # Entraînement du modèle
    history = model.fit(
        train_generator,
        epochs=config.NUM_EPOCHS,
        validation_data=valid_generator
    )

    # Enregistrer le modèle
    model.save('plant_disease_model.h5')

    # Évaluer le modèle
    loss, accuracy = model.evaluate(valid_generator)
    print(f'Validation accuracy: {accuracy * 100:.2f}%')

    # Afficher les graphiques
    plot_history(history)

if __name__ == "__main__":
    train()
