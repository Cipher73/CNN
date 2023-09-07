import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skimage.filters import gaussian
from skimage.util import random_noise
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
import io

''' class for training segmentation models '''
class SegmentationModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {
            'ResNet-34': 'saved_models/resnet34_model_WN.h5',
            'ResNet-50': 'saved_models/resnet50_model_WN.h5',
            'VGG-16': 'saved_models/vgg16_model_WN.h5'
        }

    ''' Function to build a segmentation model using given backbone '''
    def build_segmentation_model(self, backbone, n_classes, activation):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])  # Use the learning rate from config
        model = sm.Unet(
            backbone, encoder_weights='imagenet', encoder_freeze=True, classes=n_classes, activation=activation
        )
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=self.config['metrics'])
        return model

    ''' Function to train a model on given data '''
    def train_model(self, model, X_train, y_train_cat, X_val, y_val_cat):
        history = model.fit(
            X_train, y_train_cat, batch_size=self.config['batch_size'],
            epochs=self.config['num_epochs'], verbose=1, validation_data=(X_val, y_val_cat),
            callbacks=[WandbCallback()]
        )
        return history

    ''' Function to save a trained model '''
    def save_model(self, model, model_name):
        model.save(f'saved_models/{model_name}_WN.h5')

    ''' Train a single model using provided backbone '''
    def train_single_model(self, X_train, y_train_cat, X_val, y_val_cat, backbone):
        model = self.build_segmentation_model(backbone, self.config['n_classes'], self.config['activation'])
        history = self.train_model(model, X_train, y_train_cat, X_val, y_val_cat)
        self.save_model(model, f'{backbone}_model')
        self.log_metrics_to_table(history, backbone)

    ''' Train segmentation models for multiple backbones '''
    def train_segmentation_models(self, X_train, y_train_cat, X_val, y_val_cat):
        for backbone in self.config['backbones']:
            with wandb.init(
                project=self.config['wandb_project'],
                config=self.config,
                name=f"{backbone}_BS_{self.config['batch_size']}_LR_{self.config['learning_rate']}_WOD_WN"
            ):
                self.train_single_model(X_train, y_train_cat, X_val, y_val_cat, backbone)

    ''' Log metrics to WandB table '''
    def log_metrics_to_table(self, history, backbone):
        # Define a color for each backbone (or any logic you choose)
        colors = {
            'resnet34': 'red',
            'vgg16': 'blue',
            'resnet50': 'green',
        }
        stroke_color = colors.get(backbone, 'black')  # default to black if backbone not found

        # Initialize tables for different metrics
        loss_table = wandb.Table(columns=["epoch", "loss", "stroke"])
        val_loss_table = wandb.Table(columns=["epoch", "val_loss", "stroke"])
        iou_score_table = wandb.Table(columns=["epoch", "iou_score", "stroke"])
        val_iou_score_table = wandb.Table(columns=["epoch", "val_iou_score", "stroke"])

        # Loop through epochs and add data to tables
        for epoch in range(self.config['num_epochs']):
            loss_table.add_data(epoch, history.history["loss"][epoch], stroke_color)
            val_loss_table.add_data(epoch, history.history["val_loss"][epoch], stroke_color)
            iou_score_table.add_data(epoch, history.history["iou_score"][epoch], stroke_color)
            val_iou_score_table.add_data(epoch, history.history["val_iou_score"][epoch], stroke_color)

        # Log tables as plots in WandB
        wandb.log({
            "Loss": wandb.plot.line(loss_table, "epoch", "loss", stroke="stroke", title="Loss over Epochs"),
            "Val Loss": wandb.plot.line(val_loss_table, "epoch", "val_loss", stroke="stroke", title="Validation Loss over Epochs"),
            "IoU Score": wandb.plot.line(iou_score_table, "epoch", "iou_score", stroke="stroke", title="IoU Score over Epochs"),
            "Val IoU Score": wandb.plot.line(val_iou_score_table, "epoch", "val_iou_score", stroke="stroke", title="Validation IoU Score over Epochs")
        })

    ''' Evaluate trained models on test data '''
    def evaluate_models_on_test(self, X_test, y_test_cat):
        evaluation_results = {}
        accuracy_results = {}  # To accumulate accuracy values
        
        # Loop through saved models and evaluate each one
        for model_name, model_path in self.models.items():
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer=self.config['optimizer'],
                          loss='categorical_crossentropy', metrics=self.config['metrics'])
            
            # Evaluate the model
            loss = model.evaluate(X_test, y_test_cat, verbose=1)
            
            # Make predictions and calculate confusion matrix
            y_pred = model.predict(X_test)
            y_pred_argmax = np.argmax(y_pred, axis=3)
            y_test_argmax = np.argmax(y_test_cat, axis=3)
            cm = confusion_matrix(y_test_argmax.flatten(), y_pred_argmax.flatten())

            # Normalize the confusion matrix
            cm_normalized = cm / cm.sum(axis=1)

            # Plot and save the confusion matrix as an image
            plt.imshow(cm_normalized, cmap='hot')
            plt.colorbar()  # Add a color bar to indicate values
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Normalized Confusion Matrix')
            plt.savefig(f"plots/{model_name}_confusion_matrix_WN.png")
            plt.close()

            # Log confusion matrix
            wandb.init(
                project=self.config['wandb_project'],
                config=self.config,
                name=f"accuracy_{model_name}_BS_{self.config['batch_size']}_LR_{self.config['learning_rate']}_WOD_WN"
            )

            # Log loss
            wandb.log({f"{model_name}_test_loss": loss[0]})

            # Loop through some test images for visualization and logging
            for i in range(min(3, len(X_test))):
                # Clean up: Delete the temporary images if they are no longer needed
                # Save image ground truth and prediction on the same image
                plt.subplot(2,3,1)
                plt.imshow(X_test[i])
                plt.title('Image')
                plt.subplot(2,3,2)
                plt.imshow(y_test_argmax[i])
                plt.title('Ground Truth')
                plt.subplot(2,3,3)
                plt.imshow(y_pred_argmax[i])
                plt.title('Prediction')
                plt.savefig(f"plots/{model_name}_prediction_{i}_WN.png")
                plt.close()
                wandb.log({f"{model_name}_prediction_{i}": wandb.Image(f"plots/{model_name}_prediction_{i}_WN.png")})
                
            # Calculate accuracy
            accuracy = accuracy_score(y_test_argmax.flatten(), y_pred_argmax.flatten())
            accuracy_results[model_name] = accuracy
            wandb.run.finish()

        return accuracy_results
        
# Define a class for data preprocessing
class DataProcessor:
    @staticmethod
    def get_data(directory_path, flag):
        images = []
        for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
            img = cv2.imread(img_path, flag)    
            images.append(img)
        images = np.array(images)
        return images

    @staticmethod
    def shuffle_data(images, masks):
        images, masks = shuffle(images, masks, random_state=0)
        return images, masks
    
    # Preprocess data
    def preprocess_data(self, train_images, train_masks, val_images, val_masks, test_images, test_masks):
        train_images = np.array(train_images)
        train_masks = np.array(train_masks)
        val_images = np.array(val_images)
        val_masks = np.array(val_masks)
        test_images = np.array(test_images)
        test_masks = np.array(test_masks)
        
        # Label encoding for training masks
        labelencoder = LabelEncoder()
        n, h, w = train_masks.shape
        train_masks_reshaped = train_masks.reshape(-1,1)
        train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
        train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
        
        train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
        X_train = train_images
        y_train = train_masks_input
        X_test = test_images
        y_test = test_masks
        X_val = val_images
        y_val = val_masks

        n_classes = len(np.unique(y_train))
        train_masks_cat = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
        y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
        test_masks_cat = tf.keras.utils.to_categorical(y_test, num_classes=n_classes)
        y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
        val_masks_cat = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)
        y_val_cat = val_masks_cat.reshape((y_val.shape[0], y_val.shape[1], y_val.shape[2], n_classes))

        return X_train, y_train_cat, X_test, y_test_cat, X_val, y_val_cat, n_classes
 

def main():
    data_processor = DataProcessor()

    # Load and preprocess data
    train_images = np.array(data_processor.get_data("Dataset/train/images/", 1))
    train_masks = np.array(data_processor.get_data("Dataset/train/masks/", 0))
    val_images = np.array(data_processor.get_data("Dataset/val/images/", 1))
    val_masks = np.array(data_processor.get_data("Dataset/val/masks/", 0))
    test_images = np.array(data_processor.get_data("Dataset/test/noisy_images/", 1))
    test_masks = np.array(data_processor.get_data("Dataset/test/noisy_masks/", 0))

    # Shuffle the data
    train_images, train_masks = data_processor.shuffle_data(train_images, train_masks)
    val_images, val_masks = data_processor.shuffle_data(val_images, val_masks)
    test_images, test_masks = data_processor.shuffle_data(test_images, test_masks)

    # Preprocess the data
    X_train, y_train_cat, X_test, y_test_cat, X_val, y_val_cat, n_classes = data_processor.preprocess_data(
        train_images, train_masks, val_images, val_masks, test_images, test_masks
    )

    # Define configuration parameters
    learning_rate = 0.001
    config = {
        'wandb_project': "SSOCTLRE",
        'num_epochs': 10,
        'batch_size': 16,
        'backbones': ['resnet34', 'resnet50', 'vgg16'],
        'n_classes': n_classes,
        'activation': 'softmax',
        'learning_rate': learning_rate,
        'optimizer': "adam",
        'metrics': [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), sm.metrics.Precision(threshold=0.5), sm.metrics.Recall(threshold=0.5)]
    }

    # Initialize and use the SegmentationModelTrainer class
    trainer = SegmentationModelTrainer(config)
    trainer.train_segmentation_models(X_train, y_train_cat, X_val, y_val_cat)

    # Evaluate trained models on test data
    evaluation_results = trainer.evaluate_models_on_test(X_test, y_test_cat)

    for model_name, scores in evaluation_results.items():
        print(f"{model_name} Test Scores:", scores)


if __name__ == "__main__":
    main()
