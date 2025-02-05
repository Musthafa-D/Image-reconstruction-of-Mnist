import matplotlib.pyplot as plt
import os
from ccbdl.utils.logging import get_logger
from ccbdl.evaluation.plotting.base import GenericPlot
import numpy as np
import seaborn as sns
import torch
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from captum.attr import visualization as viz
from ccbdl.config_loader.loaders import ConfigurationLoader
from matplotlib.colors import LinearSegmentedColormap
from utils import load_classifier, load_gan, ImageTensorDataset, load_cgan
from utils import attribution_maps_discriminator, attribution_maps_classifier
from sklearn.metrics import confusion_matrix
from ignite.metrics import PSNR, SSIM
from ccbdl.evaluation.plotting import images
from ccbdl.network.nlrl import NLRL_AO
import math


class Decode_plot(GenericPlot):
    def __init__(self, learner):
        """
        init method of the image generation plots.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Decode_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("decoded images the model for certain epochs")
        
        self.style = {}

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def grid_2d(self, imgs, de_imgs, labels=None, figsize = (10, 10)):
        """
        Function to create of reconstructions of img data.

        Parameters
        ----------
        imgs (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].      
        figsize (tuple of ints, optional): Size of the figure in inches. Defaults to (10, 10).

        Returns
        -------
        figs of the grid images.

        """
        # try to make a square of the img.
        if not len(imgs.shape) == 4:
            # unexpected shapes
            pass
        num = len(imgs)
        rows = int(num / math.floor((num)**0.5))
        cols = int(math.ceil(num/rows))
        
        fig = plt.figure(figsize = figsize)
        for i in range(num):
            ax1 = plt.subplot(rows, 2 * cols, 2 * i + 1)
            ax2 = plt.subplot(rows, 2 * cols, 2 * i + 2)

            # Get contents
            img = imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
            de_img = de_imgs[i].cpu().detach().permute(1, 2, 0).squeeze()
            
            if labels is not None:
                label = labels[i]

            # Plot noise
            ax1.imshow(img, cmap='gray')
            if labels is not None:
                ax1.set_title(f"{i}\nlable: {label}")
            else:
                ax1.set_title(f"{i}")
            
            ax2.imshow(de_img, cmap='gray')
            ax2.set_title(f"{i}\ndecoded")

            # Remove axes
            ax1.axis('off')
            ax2.axis('off')

        plt.tight_layout()
        
        return fig

    def plot(self):
        """
        method to plot the generated images' plots.

        Returns
        -------
        figs of the decoded images' plots, names of the decoded images' plots (for saving the plots with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")      
        total = len(epochs)

        figs = []
        names = []
        
        for types in ["train", "test"]:
            if types == "train":
                original_images = self.learner.data_storage.get_item("real_images")
                fake_images = self.learner.data_storage.get_item("fake_images")
                if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                    labels = self.learner.data_storage.get_item("labels")
            else:
                original_images = self.learner.data_storage.get_item("real_images_test")
                fake_images = self.learner.data_storage.get_item("fake_images_test")
                if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                    labels = self.learner.data_storage.get_item("labels_test")
            
            # Number of batches per epoch
            batches_per_epoch = int(len(original_images)/total)
            
            for idx in range(total):
                # Calculate the index for the last batch of the current epoch
                last_batch_index = ((idx + 1) * batches_per_epoch) - 1
                
                original_images_per_epoch = original_images[last_batch_index]
                fake_images_per_epoch = fake_images[last_batch_index]
                if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                    labels_per_epoch = labels[last_batch_index]
                    
                num = original_images_per_epoch.size(0)
                
                self.style["figsize"] = self.get_value_with_default("figsize", (num, num), {"figsize": (20, 20)})
                
                epoch = epochs[idx]
                
                if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                    figs.append(self.grid_2d(imgs=original_images_per_epoch,
                                             de_imgs=fake_images_per_epoch,
                                             labels=labels_per_epoch,
                                               **self.style))
                else:
                    figs.append(self.grid_2d(imgs=original_images_per_epoch,
                                             de_imgs=fake_images_per_epoch,
                                               **self.style))
                
                if types == "train":
                    names.append(os.path.join("decoded_images", "train", f"epoch_{epoch}"))
                else:
                    names.append(os.path.join("decoded_images", "test", f"epoch_{epoch}"))
            
        return figs, names
        

class Loss_plot(GenericPlot):
    def __init__(self, learner):
        """
        init method of the loss plot.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Loss_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating loss plot")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def plot(self):
        """
        method to plot the loss plots.

        Returns
        -------
        fig of the loss plot, name of the loss plot (for saving the plot with this name).

        """
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_loss")
        yatr = self.learner.data_storage.get_item("a_train_loss")
        yt = self.learner.data_storage.get_item("test_loss")

        figs = []
        names = []
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

        # First subplot with logarithmic scale
        ax1.plot(x, ytr, label='train')
        ax1.plot(x, yatr, label='train_avg')
        ax1.plot(x, yt, label='test')
        ax1.set_xlabel("batch")
        ax1.set_ylabel("loss")
        ax1.set_title("loss vs batch")
        ax1.grid(True)
        ax1.legend()

        # Second subplot with a zoomed-in view
        ax2.plot(x, ytr, label='train')
        ax2.plot(x, yatr, label='train_avg')
        ax2.plot(x, yt, label='test')
        ax2.set_xlabel("batch")
        ax2.set_ylabel("loss")
        ax2.set_title("loss vs batch")
        ax2.set_ylim([0, 0.2])
        ax2.grid(True)
        ax2.legend()
        
        fig.tight_layout()
        figs.append(fig)
        names.append(os.path.join("losses"))

        return figs, names


class TimePlot(GenericPlot):
    def __init__(self, learner):
        """
        init method of the time plot.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(TimePlot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating time plot")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def plot(self):
        """
        method to plot the time plot

        Returns
        -------
        fig of the time plot, name of the time plot (for saving the plot with this name).

        """
        figs = []
        names = []
        fig, ax = plt.subplots(figsize=(12, 8))
        
        xs, ys = zip(*self.learner.data_storage.get_item("Time", batch=True))
        
        ax.plot(xs, [y - ys[0]for y in ys], label="train_time")
        ax.set_xlabel("batch")
        ax.set_ylabel("time")
        ax.set_title("time vs batch")
        ax.legend()
        
        figs.append(fig)
        names.append(os.path.join("time_plot"))
        
        return figs, names


class Fid_plot(GenericPlot):
    def __init__(self, learner):
        super(Fid_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating fid score plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        y = self.learner.data_storage.get_item("fid_score")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), gridspec_kw={'height_ratios': [2, 1]})

        # First subplot with logarithmic scale
        ax1.plot(x, y, label="CNN_RGB")
        ax1.set_xlabel("batch")
        ax1.set_ylabel("fid score")
        ax1.set_title("fid score vs batch")
        ax1.grid(True)
        ax1.legend()

        # Second subplot with a zoomed-in view
        ax2.plot(x, y, label="CNN_RGB")
        ax2.set_xlabel("batch")
        ax2.set_ylabel("fid score")
        ax2.set_title("fid score vs batch (Zoomed In)")
        ax2.set_ylim([0, 1])
        ax2.grid(True)
        ax2.legend()

        fig.tight_layout()
        figs = [fig]
        names = [os.path.join("fid_scores")]

        return figs, names


class Psnr_plot(GenericPlot):
    def __init__(self, learner):
        """
        init method of the psnr plot.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Psnr_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating psnr plot")
        
        self.psnr_scores = []
        # Initialize the PSNR metric from ignite
        self.psnr_metric = PSNR(data_range=1.0, device=learner.device)

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def calculate_psnr_score(self, real_images, generated_images, epochs):
        """
        Method to calculate the psnr scores using real and generated images.

        Parameters
        ----------
        real_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W] from the dataset.      
        generated_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].        
        epochs : number of epochs run.

        Returns
        -------
        psnr scores.

        """
        for epoch in epochs:
            # Resetting the PSNR metric for each epoch
            self.psnr_metric.reset()
            # Update the metric with real and generated images
            self.psnr_metric.update((generated_images[epoch], real_images[epoch]))
            # Compute the PSNR for the current epoch
            psnr = self.psnr_metric.compute()
            self.psnr_scores.append(psnr)
        
        return self.psnr_scores
    
    def plot(self):
        """
        method to plot the psnr plot

        Returns
        -------
        fig of the psnr plot, name of the psnr plot (for saving the plot with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")
        real_images = self.learner.data_storage.get_item("real_images")
        generated_images = self.learner.data_storage.get_item("fake_images")
        
        psnr_scores = self.calculate_psnr_score(real_images, generated_images, epochs)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(epochs, psnr_scores, label="CNN_RGB")
        ax.set_xlabel("epochs")
        ax.set_ylabel("psnr score")
        ax.set_xticks(range(len(epochs)))
        ax.set_title("psnr score vs epoch")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs = [fig]
        names = [os.path.join("psnr_scores")]

        return figs, names


class Ssim_plot(GenericPlot):
    def __init__(self, learner):
        """
        init method of the ssim plot.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Ssim_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating ssim plot")
        
        self.ssim_metric = SSIM(data_range=1.0, device=learner.device) # Initialize the SSIM metric from ignite
        self.ssim_scores = []
    
    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def calculate_ssim_score(self, real_images, generated_images, epochs):
        """
        Method to calculate the ssim scores using real and generated images.

        Parameters
        ----------
        real_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W] from the dataset.      
        generated_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].      
        epochs : number of epochs run.

        Returns
        -------
        ssim scores.

        """
        for epoch in epochs:
            # Reset the SSIM metric for each epoch
            self.ssim_metric.reset()
            # Update the metric with the real and generated images
            self.ssim_metric.update((generated_images[epoch], real_images[epoch]))
            # Compute the SSIM for the current batch
            ssim = self.ssim_metric.compute()
            self.ssim_scores.append(ssim)
        
        return self.ssim_scores

    def plot(self):
        """
        method to plot the ssim plot

        Returns
        -------
        fig of the ssim plot, name of the ssim plot (for saving the plot with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")
        real_images = self.learner.data_storage.get_item("real_images")
        generated_images = self.learner.data_storage.get_item("fake_images")
        
        ssim_scores = self.calculate_ssim_score(real_images, generated_images, epochs)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(epochs, ssim_scores, label="CNN_RGB")
        ax.set_xlabel("epochs")
        ax.set_ylabel("ssim score")
        ax.set_xticks(range(len(epochs)))
        ax.set_title("ssim score vs epoch")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        figs = [fig]
        names = [os.path.join("ssim")]

        return figs, names


class Confusion_matrix_classifier(GenericPlot):
    def __init__(self, learner, **kwargs):
        """
        init method of the confusion matrix plots with respect to pre trained classifier.

        Parameters
        ----------
        learner : learner class.      
        **kwargs (TYPE): Additional Parameters for further tasks.

        Returns
        -------
        None.

        """
        super(Confusion_matrix_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating confusion matrix based on pre trained classifier")
        
        # get default stype values
        self.style = {}
        self.style["figsize"] = self.get_value_with_default("figsize", (12, 12), kwargs)
        self.style["cmap"] = self.get_value_with_default("cmap", "Blues", kwargs)
        self.style["ticks"] = self.get_value_with_default("ticks", "auto", kwargs)
        self.style["xrotation"] = self.get_value_with_default("xrotation", "vertical", kwargs)
        self.style["yrotation"] = self.get_value_with_default("yrotation", "horizontal", kwargs)
        self.style["color_threshold"] = self.get_value_with_default("color_threshold", 50, kwargs)
        self.style["color_high"] = self.get_value_with_default("color_high", "white", kwargs)
        self.style["color_low"] = self.get_value_with_default("color_low", "black", kwargs)
    
    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def plot(self):
        """
        method to plot the confusion matrix plots.

        Returns
        -------
        figs of the confusion matrix plots, names of the confusion matrix plots (for saving the plots with this name).

        """
        process_imgs = Tsne_plot_classifier(self.learner)
        
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the classifier
            classifier = load_classifier(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the classifier
            classifier = load_classifier(layer='nlrl')
        else:
                raise ValueError("Invalid values, it's either linear or nlrl")
        
        # Setting concatenation false by initializing value as 0
        cat = 0
        
        names = []
        figs = []
        
        total_real_images = self.learner.data_storage.get_item("real_images")
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        total_true_labels = self.learner.data_storage.get_item("labels")
        epochs = self.learner.data_storage.get_item("epochs_gen")
        
        total = len(epochs)
        # Number of batches per epoch
        batches_per_epoch = int(len(total_real_images)/total)
        
        for idx in range(total):
            epoch = epochs[idx]
            
            # Calculate the index for the last batch of the current epoch
            last_batch_index = ((idx + 1) * batches_per_epoch) - 1
            
            # Access the last batch of real, fake, and label data for the current epoch
            real_images = total_real_images[last_batch_index]
            fake_images = total_fake_images[last_batch_index]
            true_labels = total_true_labels[last_batch_index]
    
            # Process real images
            real_dataset = ImageTensorDataset(real_images)
            real_data_loader = DataLoader(real_dataset, batch_size=1, shuffle=False)
            _, real_predicted_labels = process_imgs.process_images(real_data_loader, classifier, cat)
            
            # Process fake images
            fake_dataset = ImageTensorDataset(fake_images)
            fake_data_loader = DataLoader(fake_dataset, batch_size=1, shuffle=False)
            _, fake_predicted_labels = process_imgs.process_images(fake_data_loader, classifier, cat)
            
            for types in ["real", "fake"]:
                predictions = real_predicted_labels if types == 'real' else fake_predicted_labels
                correct_labels = true_labels
                
                # Flatten the predictions list to a single tensor
                # Each tensor in the list is a single-element tensor, so we concatenate them and then flatten
                predictions_tensor = torch.cat(predictions, dim=0).flatten()
                
                # Move tensors to CPU and convert to numpy for sklearn compatibility
                predictions_np = predictions_tensor.cpu().numpy()
                correct_labels_np = correct_labels.cpu().numpy()
                
                figs.append(images.plot_confusion_matrix(predictions_np,
                                                         correct_labels_np,
                                                         **self.style))

                names.append(os.path.join("confusion_matrices", "classifier_training_based", f"{types}", f"epoch_{epoch}"))
        return figs, names


class Confusion_matrix_gan(GenericPlot):
    def __init__(self, learner):
        """
        init method of the confusion matrix plots with respect to pre trained gan or conditional.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Confusion_matrix_gan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("Creating confusion matrix for real vs. fake predictions")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def plot(self):
        """
        method to plot the confusion matrix plots.

        Returns
        -------
        figs of the confusion matrix plots, names of the confusion matrix plots (for saving the plots with this name).

        """
        names = []
        figs = []
        
        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")
        
        if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
            labels = self.learner.data_storage.get_item("labels")
        
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)
        threshold = 0.5
        
        # Number of batches per epoch
        batches_per_epoch = int(len(real_images)/total)
        
        total = len(epochs)
        
        if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
            if self.learner.learner_config["layer"] == 'linear': 
                # Load the model
                cgan = load_cgan(layer='linear')
            elif self.learner.learner_config["layer"] == 'nlrl':
                # Load the model
                cgan = load_cgan(layer='nlrl')
            else:
                raise ValueError("Invalid values, it's either linear or nlrl")
                
            model = cgan.discriminator
        else:
            if self.learner.learner_config["layer"] == 'linear': 
                # Load the model
                gan = load_gan(layer='linear')
            elif self.learner.learner_config["layer"] == 'nlrl':
                # Load the model
                gan = load_gan(layer='nlrl')
            else:
                raise ValueError("Invalid values, it's either linear or nlrl")

            model = gan.discriminator
        
        for idx in range(total):
            fig, ax = plt.subplots(figsize=(12, 8))
            epoch = epochs[idx]
            
            # Calculate the index for the last batch of the current epoch
            last_batch_index = ((idx + 1) * batches_per_epoch) - 1
            
            if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                real_probs = model(real_images[last_batch_index], labels[last_batch_index])
                fake_probs = model(fake_images[last_batch_index], labels[last_batch_index])
            else:
                real_probs = model(real_images[last_batch_index])
                fake_probs = model(fake_images[last_batch_index])
            
            real_probs = real_probs.detach().cpu().numpy()
            fake_probs = fake_probs.detach().cpu().numpy()
            
            # Extract and flatten the predictions for the current epoch
            fake_probs_per_epoch = fake_probs.flatten()
            real_probs_per_epoch = real_probs.flatten()

            # Convert probabilities to binary predictions using a threshold eg 0.5
            fake_predictions = (fake_probs_per_epoch < threshold).astype(int)
            real_predictions = (real_probs_per_epoch > threshold).astype(int)

            # Concatenate predictions and true labels
            predictions = np.concatenate([fake_predictions, real_predictions])
            correct_labels = np.concatenate([np.zeros_like(fake_predictions), np.ones_like(real_predictions)])
            
            # Compute confusion matrix
            matrix = confusion_matrix(correct_labels, predictions)

            # Plot the confusion matrix using seaborn's heatmap
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', ax=ax,
                        xticklabels=['Predicted Fake', 'Predicted Real'],
                        yticklabels=['Actual Fake', 'Actual Real'])
            
            figs.append(fig)
            names.append(os.path.join("confusion_matrices", "gan_training_based", f"epoch_{epoch}"))
        return figs, names


class Tsne_plot_images_combined(GenericPlot):
    def __init__(self, learner):
        """
        init method of the tsne plot.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Tsne_plot_images_combined, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating combined tsne plots")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def plot(self):
        """
        method to plot the tsne plot.

        Returns
        -------
        fig of the tnse plot, name of the tsne plot (for saving the plot with this name).

        """
        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")
        labels = self.learner.data_storage.get_item("labels")

        num_classes = 10
        # Create color map for real and fake images
        colors = cm.rainbow(np.linspace(0, 1, num_classes))

        tsne = TSNE(n_components=2)

        fig, ax = plt.subplots(figsize=(16, 10))

        # Storage for transformed data
        real_transformed_list = []
        fake_transformed_list = []

        for lb in range(num_classes):
            idx = labels == lb
            real_image = real_images[idx].view(real_images[idx].size(0), -1).cpu().numpy()  # real_images of current class
            fake_image = fake_images[idx].view(fake_images[idx].size(0), -1).cpu().numpy()  # generated_images of current class

            # Compute TSNE for each class and store
            real_transformed_list.append(tsne.fit_transform(real_image))
            fake_transformed_list.append(tsne.fit_transform(fake_image))

        # Plot each class for real and fake images
        for lb in range(num_classes):
            ax.scatter(real_transformed_list[lb][:, 0], real_transformed_list[lb][:, 1], c=[colors[lb]], label=f'{lb}', marker='o')
            ax.scatter(fake_transformed_list[lb][:, 0], fake_transformed_list[lb][:, 1], c=[colors[lb]], label=f'{lb}', marker='^')

        ax.set_title("Combined t-SNE plot for all labels")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.show()
        figs = [fig]
        names = [os.path.join("analysis_plots", "tsne_plots", "combined_tsne_plot")]
        return figs, names


class Tsne_plot_images_separate(GenericPlot):
    def __init__(self, learner):
        """
        init method of the tsne plots.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Tsne_plot_images_separate, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots for each label")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def plot(self):
        """
        method to plot the tsne plot.

        Returns
        -------
        figs of the tnse plots, names of the tsne plots (for saving the plots with this name).

        """
        real_images = self.learner.data_storage.get_item("real_images")
        fake_images = self.learner.data_storage.get_item("fake_images")
        labels = self.learner.data_storage.get_item("labels")

        figs = []
        names = []

        num_classes = 10

        for lb in range(num_classes):
            idx = labels == lb
            real_image = real_images[idx]  # real_images of current class  
            fake_image = fake_images[idx] # generated_images of current class

            real_image = real_image.view(real_image.size(0), -1).cpu().detach().numpy()
            fake_image = fake_image.view(fake_image.size(0), -1).cpu().numpy()

            # compute TSNE
            tsne = TSNE(n_components=2)

            real_transformed = tsne.fit_transform(real_image)
            fake_transformed = tsne.fit_transform(fake_image)

            fig, ax = plt.subplots(figsize=(16, 10))

            ax.scatter(
                real_transformed[:, 0], real_transformed[:, 1], label='real images', marker='o')

            ax.scatter(
                fake_transformed[:, 0], fake_transformed[:, 1], label='fake images', marker='^')

            ax.set_title(f"t-SNE plot for label {lb}")
            ax.legend()

            figs.append(fig)
            names.append(os.path.join("analysis_plots", "tsne_plots", "seperate_plots", f"label_{lb}"))          
        return figs, names

    
class Tsne_plot_classifier(GenericPlot):
    def __init__(self, learner):
        """
        init method of the tsne plots based on pre trained classifier.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Tsne_plot_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on classifier's features and classifier's decision")
    
    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def get_features(self, classifier, imgs):
        """
        method to extract the features from a layer of the pre trained classifier.

        Parameters
        ----------
        classifier : pre trained classifier.       
        imgs (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].

        Returns
        -------
        features from the classifier.

        """
        activation = {}
        
        def get_activation(name):
            def hook(classifier, inp, output):
                activation[name] = output.detach()
            return hook
        
        # Register the hook
        if self.learner.learner_config["layer"] == 'nlrl':
            handle = classifier.model[-2].register_forward_hook(get_activation('conv'))
        else:
            handle = classifier.model[-1].register_forward_hook(get_activation('conv'))
            
        _ = classifier(imgs)
        
        # Remove the hook
        handle.remove()
        return activation['conv']
    
    def compute_tsne(self, features):
        """
        method to compute the tsne with features.

        Parameters
        ----------
        features : extracted features from the classifier.

        Returns
        -------
        tsne results after computation.

        """
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(features)
        return tsne_results
    
    def process_images(self, data_loader, classifier, cat):
        """
        method to process the images from the data loader.

        Parameters
        ----------
        data_loader : the respective images' dataloader.       
        classifier : pre trained classifier.       
        cat (int): condition to check whether concatenation of features and labels required or not.

        Returns
        -------
        features and labels of the dataloader's images'.

        """
        all_features = []
        all_labels = []
        
        for imgs in data_loader:
            outputs = classifier(imgs)
            _, predicted_labels = torch.max(outputs, 1)
            features = self.get_features(classifier, imgs)
            features = features.view(features.size(0), -1)  # Flatten the features
            all_features.append(features)
            all_labels.append(predicted_labels)
            
        # Concatenate all the features and labels from the batches
        if cat == 1:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        return all_features, all_labels
    
    def plot(self):
        """
        method to plot the tsne plots.

        Returns
        -------
        figs of the tnse plots, names of the tsne plots (for saving the plots with this name).

        """
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the classifier
            classifier = load_classifier(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the classifier
            classifier = load_classifier(layer='nlrl')
        else:
            raise ValueError("Invalid values, it's either linear or nlrl")
        
        # Setting concatenation true by initializing value as 1
        cat = 1
        config = ConfigurationLoader().read_config("config.yaml")
        data_config = config["data"]
        
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)
    
        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        batches_per_epoch = int(len(total_real_images)/total)
        
        real_images = total_real_images[-batches_per_epoch:]
        real_images = torch.cat(real_images)
        real_dataset = ImageTensorDataset(real_images)
        real_data_loader = DataLoader(real_dataset, batch_size=data_config["batch_size"], shuffle=False)
        real_features, real_labels = self.process_images(real_data_loader, classifier, cat)
    
        # Process generated images  
        total_generated_images = self.learner.data_storage.get_item("fake_images")
        
        generated_images = total_generated_images[-batches_per_epoch:]
        generated_images = torch.cat(generated_images)
        generated_dataset = ImageTensorDataset(generated_images)
        generated_data_loader = DataLoader(generated_dataset, batch_size=data_config["batch_size"], shuffle=False)
        generated_features, generated_labels = self.process_images(generated_data_loader, classifier, cat)
        
        real_label_counts = [torch.sum(real_labels == i).item() for i in range(10)]
        fake_label_counts = [torch.sum(generated_labels == i).item() for i in range(10)]
    
        # Combine features for t-SNE
        combined_features = torch.cat([real_features, generated_features], dim=0)
        tsne_results = self.compute_tsne(combined_features.cpu().numpy())
    
        # Split t-SNE results back into real and generated
        real_tsne = tsne_results[:len(real_features)]
        fake_tsne = tsne_results[len(real_features):]
    
        # Plotting
        figs, names = [], []
        for label in range(10):  # Mnist dataset has 10 labels
            fig, ax = plt.subplots(figsize=(16, 10))
            # Real images scatter plot
            real_indices = (real_labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=real_tsne[real_indices, 0], 
                y=real_tsne[real_indices, 1], 
                label=f"Real {label}",
                marker="o",
                alpha=0.5,
                color='darkred'
            )
            # Fake images scatter plot
            fake_indices = (generated_labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=fake_tsne[fake_indices, 0], 
                y=fake_tsne[fake_indices, 1], 
                label=f"Fake {label}", 
                marker="^",
                alpha=0.5,
                color='black'
            )
            ax.set_title(f"t-SNE visualization for label {label}, Counts - Real: {real_label_counts[label]}, Fake: {fake_label_counts[label]}")
            ax.legend()
            figs.append(fig)
            names.append(os.path.join("analysis_plots", "tsne_plots", "feature_based_classifier", f"label_{label}"))   
        return figs, names


class Tsne_plot_dis_gan(GenericPlot):
    def __init__(self, learner):
        """
        init method of the tsne plots based on pre trained discriminator.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Tsne_plot_dis_gan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on gan's discriminator's features and discriminator's decision")
    
    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def get_features(self, layer_num, real_images, fake_images, discriminator):
        """
        method to extract the features from a layer of the pre trained discriminator.

        Parameters
        ----------
        layer_num (int): specifying the layer number of the discriminator from which the features are extracted.     
        real_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W] from the dataset.      
        fake_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].       
        discriminator : pre trained discriminator.

        Returns
        -------
        features from the discriminator.

        """
        features = []

        def hook(discriminator, inp, output):
            features.append(output.detach())

        # Attach the hook to the desired layer
        handle = discriminator[layer_num].register_forward_hook(hook)

        # Process real images through the discriminator
        for imgs in real_images:
            discriminator(imgs)

        # Process fake images through the discriminator
        for imgs in fake_images:
            discriminator(imgs)

        handle.remove()  # Remove the hook
        return torch.cat(features)
    
    def compute_tsne(self, features):
        """
        method to compute the tsne with features.

        Parameters
        ----------
        features : extracted features from the discriminator.

        Returns
        -------
        tsne results after computation.

        """
        # Compute t-SNE embeddings from features
        tsne = TSNE(n_components=2, random_state=0)
        return tsne.fit_transform(features.cpu().numpy())
    
    def plot(self):
        """
        method to plot the tsne plots.

        Returns
        -------
        figs of the tnse plots, names of the tsne plots (for saving the plots with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)
        
        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        batches_per_epoch = int(len(total_real_images)/total)
        real_images = total_real_images[-batches_per_epoch:]
        
        # Process fake images
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        fake_images = total_fake_images[-batches_per_epoch:]
        
        # Process labels
        total_labels = self.learner.data_storage.get_item("labels")
        labels = total_labels[-batches_per_epoch:]
        
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the model
            gan = load_gan(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the model
            gan = load_gan(layer='nlrl')
        else:
            raise ValueError("Invalid values, it's either linear or nlrl")

        discriminator = gan.discriminator
        
        if self.learner.learner_config["layer"] == 'nlrl':
            layer_num = -4
        else:
            layer_num = -4
        
        # Extract features from the discriminator
        features = self.get_features(layer_num, real_images, fake_images, discriminator)
        # Flatten the features
        features = features.view(features.size(0), -1)
        
        labels = torch.cat(labels, dim=0)
        
        # Compute t-SNE
        tsne_results = self.compute_tsne(features)
        
        half = len(tsne_results) // 2

        # Split t-SNE results back into real and fake
        real_tsne = tsne_results[:half]
        fake_tsne = tsne_results[half:]
        
        # Plotting
        figs, names = [], []
        
        # Plotting
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot real images with different colors for each label
        for label in range(10):
            label_indices = (labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=real_tsne[label_indices, 0], 
                y=real_tsne[label_indices, 1], 
                label=f"Real Images (Label {label})", 
                alpha=0.5,
                marker="o"
                
            )

        # Plot fake images
        sns.scatterplot(
            ax=ax, 
            x=fake_tsne[:, 0], 
            y=fake_tsne[:, 1], 
            label="Fake Images", 
            alpha=0.5, 
            marker="^",
            color='black'
        )

        ax.set_title("t-SNE visualization of Real and Fake Images")
        ax.legend()

        # Saving the figure
        fig_name = os.path.join("analysis_plots", "tsne_plots", "feature_based_gan_discriminator", "combined_plot")
        figs, names = [fig], [fig_name]
        return figs, names


class Tsne_plot_dis_cgan(GenericPlot):
    def __init__(self, learner):
        """
        init method of the tsne plots based on pre trained conditional discriminator.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Tsne_plot_dis_cgan, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on cgan's discriminator's features and discriminator's decision")
    
    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True
    
    def get_features(self, layer_num, real_images, fake_images, discriminator, labels):
        """
        method to extract the features from a layer of the pre trained conditional discriminator.

        Parameters
        ----------
        layer_num (int): specifying the layer number of the discriminator from which the features are extracted.    
        real_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W] from the dataset.    
        fake_images (torch.Tensor): Tensor with N imgs with the shape [N x C x H x W].   
        discriminator : pre trained conditional discriminator.

        Returns
        -------
        features from the conditional discriminator.

        """
        features = []

        def hook(discriminator, inp, output):
            features.append(output.detach())

        # Attach the hook to the desired layer
        handle = discriminator.dis[layer_num].register_forward_hook(hook)

        # Process real images through the discriminator
        for imgs, lbls in zip(real_images, labels):
            discriminator(imgs, lbls)

        # Process fake images through the discriminator
        for imgs, lbls in zip(fake_images, labels):
            discriminator(imgs, lbls)

        handle.remove()  # Remove the hook
        return torch.cat(features)
    
    def compute_tsne(self, features):
        """
        method to compute the tsne with features.

        Parameters
        ----------
        features : extracted features from the conditional discriminator.

        Returns
        -------
        tsne results after computation.

        """
        # Compute t-SNE embeddings from features
        tsne = TSNE(n_components=2, random_state=0)
        return tsne.fit_transform(features.cpu().numpy())
    
    def plot(self):
        """
        method to plot the tsne plots.

        Returns
        -------
        figs of the tnse plots, names of the tsne plots (for saving the plots with this name).

        """
        epochs = self.learner.data_storage.get_item("epochs_gen")
        total = len(epochs)
        
        # Process real images
        total_real_images = self.learner.data_storage.get_item("real_images")
        batches_per_epoch = int(len(total_real_images)/total)
        real_images = total_real_images[-batches_per_epoch:]
        
        # Process fake images
        total_fake_images = self.learner.data_storage.get_item("fake_images")
        fake_images = total_fake_images[-batches_per_epoch:]
        
        # Process labels
        total_labels = self.learner.data_storage.get_item("labels")
        labels = total_labels[-batches_per_epoch:]
        
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the model
            cgan = load_cgan(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the model
            cgan = load_cgan(layer='nlrl')
        else:
            raise ValueError("Invalid values, it's either linear or nlrl")
            
        discriminator = cgan.discriminator 
        
        if self.learner.learner_config["layer"] == 'nlrl':
            layer_num = -4
        else:
            layer_num = -4
           
        # Extract features from the discriminator
        features = self.get_features(layer_num, real_images, fake_images, discriminator, labels)
        # Flatten the features
        features = features.view(features.size(0), -1)
        
        labels = torch.cat(labels, dim=0)
        label_counts = [torch.sum(labels == i).item() for i in range(10)]
        
        # Compute t-SNE
        tsne_results = self.compute_tsne(features)
        
        half = len(tsne_results) // 2

        # Split t-SNE results back into real and fake
        real_tsne = tsne_results[:half]
        fake_tsne = tsne_results[half:]
        
        # Plotting
        figs, names = [], []
        for label in range(10):  # Mnist dataset has 10 labels
            fig, ax = plt.subplots(figsize=(16, 10))
            # Real images scatter plot
            real_indices = (labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=real_tsne[real_indices, 0], 
                y=real_tsne[real_indices, 1], 
                label=f"Real {label}", 
                alpha=0.5,
                marker="o"
            )
            # Fake images scatter plot
            fake_indices = (labels == label).cpu().numpy()
            sns.scatterplot(
                ax=ax, 
                x=fake_tsne[fake_indices, 0], 
                y=fake_tsne[fake_indices, 1], 
                label=f"Fake {label}", 
                alpha=0.5,
                marker="^",
            )
            ax.set_title(f"t-SNE visualization for label {label}, count: {label_counts[label]}")
            ax.legend()
            figs.append(fig)
            names.append(os.path.join("analysis_plots", "tsne_plots", "feature_based_cgan_discriminator", f"label_{label}"))   
        return figs, names


class Attribution_plots_classifier(GenericPlot):
    def __init__(self, learner):
        """
        init method of the attribution plots based on pre trained classifier

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Attribution_plots_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps for grayscale images")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def safe_visualize(self, attr, original_image, title, fig, ax, label, img_name, types, cmap):
        """
        method to ensure safe visualization of the attribution maps

        Parameters
        ----------
        attr : respective attribution map.
        original_image (torch.Tensor): Tensor with 1 img with the shape [C x H x W]
        title (str): title of the attribution map.
        fig : fig of the respective attribution maps.
        ax : axis of the respective attribution maps.
        label : label of the respective original_image.
        img_name : label name of the respective original image.
        types (str): real or fake representation of the images.
        cmap : cmap of gray or any custom colour.

        Returns
        -------
        None.

        """
        if not (attr == 0).all():
            if len(attr.shape) == 2:
                attr = np.expand_dims(attr, axis=2)
            if original_image is not None and len(original_image.shape) == 2:
                original_image = np.expand_dims(original_image, axis=2)
            # Call the visualization function with the squeezed attribute
            viz.visualize_image_attr(attr,
                                     original_image=np.squeeze(original_image),  # Squeeze the image as well
                                     method='heat_map',
                                     sign='all',
                                     show_colorbar=True,
                                     title=title,
                                     plt_fig_axis=(fig, ax),
                                     cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data's label: {label}, {img_name} for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        """
        method to plot the attribution plots.

        Returns
        -------
        figs of the attribution plots, names of the attribution plots (for saving the plots with this name).

        """
        names = []
        figs = []
        max_images_per_plot = 4  # Define a constant for the maximum number of images per plot
        
        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")
        
        class_dict = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
                          6: "6", 7: "7", 8: "8", 9: "9"}
        
        if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
            labels_list = self.learner.data_storage.get_item("labels")
        
        if self.learner.learner_config["layer"] == 'linear': 
            # Load the model
            model = load_classifier(layer='linear')
        elif self.learner.learner_config["layer"] == 'nlrl':
            # Load the model
            model = load_classifier(layer='nlrl')
        else:
            raise ValueError("Invalid values, it's either linear or nlrl")
                
        cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])

        for types in ["real", "fake"]:
            inputs = real_images[-1].clone() if types == 'real' else fake_images[-1].clone()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                labels = labels_list[-1].clone()
            
            inputs.requires_grad = True  # Requires gradients set true

            # Get attribution maps for different techniques
            saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps_classifier(model, 
                                                                                                                                  inputs, 
                                                                                                                                  preds)

            attrs = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution", "Occlusion"]
            
            # Process all input images
            total_indices = list(range(inputs.shape[0]))
            chunks = [total_indices[x:x + max_images_per_plot] for x in range(0, len(total_indices), max_images_per_plot)]

            for chunk in chunks:
                num_rows = len(chunk)
                num_cols = len(attrs) + 1  # One column for the original image, one for each attribution method
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
            
                # Adjust the shape of axs array if needed
                if num_rows == 1 and num_cols == 1:
                    axs = np.array([[axs]])
                elif num_rows == 1:
                    axs = axs[np.newaxis, :]
                elif num_cols == 1:
                    axs = axs[:, np.newaxis]
            
                count = 0
                for idx in chunk:
                    img = np.squeeze(inputs[idx].cpu().detach().numpy())  # Squeeze the image to 2D
                    pred = preds[idx]
                    if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                        label = labels[idx]
            
                    # Retrieve the attribution maps for the current image
                    results = [
                        np.squeeze(saliency_maps[idx].cpu().detach().numpy()),
                        np.squeeze(guided_backprop_maps[idx].cpu().detach().numpy()),
                        np.squeeze(input_x_gradient_maps[idx].cpu().detach().numpy()),
                        np.squeeze(deconv_maps[idx].cpu().detach().numpy()),
                        np.squeeze(occlusion_maps[idx].cpu().detach().numpy())
                    ]
            
                    # Display the original image
                    axs[count, 0].imshow(img, cmap='gray')
                    if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                        axs[count, 0].set_title(f"Actual labels: {label},\nPredicted labels: {pred}")
                    else:
                        axs[count, 0].set_title(f"Predicted labels: {pred}")
                    axs[count, 0].axis("off")
            
                    # Display each of the attribution maps next to the original image
                    for col, (attr, res) in enumerate(zip(attrs, results)):
                        title = f"{attr}"
                        
                        if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                            self.safe_visualize(res, img, title, fig, axs[count, col + 1], label, class_dict[label.item()], types, cmap)
                        else:
                            # Call the visualization function, passing None for label and img_name since they are not applicable
                            self.safe_visualize(res, img, title, fig, axs[count, col + 1], None, None, types, cmap)
            
                    count += 1
            
                # Set the overall title for the figure based on the type of data
                fig.suptitle(f"Classifier's view on {types.capitalize()} Data and the respective Attribution maps")
            
                # Store the figure with an appropriate name
                figs.append(fig)
                names.append(os.path.join("analysis_plots", "attribution_plots_classifier", f"{types}_data", f"chunks_{chunks.index(chunk) + 1}"))
        return figs, names


class Attribution_plots_discriminator(GenericPlot):
    def __init__(self, learner):
        """
        init method of the attribution plots based on pre trained discriminator or conditional discriminator.

        Parameters
        ----------
        learner : learner class.

        Returns
        -------
        None.

        """
        super(Attribution_plots_discriminator, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps for some grayscale images of mnist")

    def consistency_check(self):
        """
        method to check consistency.

        Returns
        -------
        bool: Always True.

        """
        return True

    def safe_visualize(self, attr, original_image, title, fig, ax, label, img_name, types, cmap):
        """
        method to ensure safe visualization of the attribution maps

        Parameters
        ----------
        attr : respective attribution map.
        original_image (torch.Tensor): Tensor with 1 img with the shape [C x H x W]
        title (str): title of the attribution map.
        fig : fig of the respective attribution maps.
        ax : axis of the respective attribution maps.
        label : label of the respective original_image.
        img_name : label name of the respective original image.
        types (str): real or fake representation of the images.
        cmap : cmap of gray or any custom colour.

        Returns
        -------
        None.

        """
        if not (attr == 0).all():
            if len(attr.shape) == 2:
                attr = np.expand_dims(attr, axis=2)
            if original_image is not None and len(original_image.shape) == 2:
                original_image = np.expand_dims(original_image, axis=2)
            # Call the visualization function with the squeezed attribute
            viz.visualize_image_attr(attr,
                                     original_image=np.squeeze(original_image),  # Squeeze the image as well
                                     method='heat_map',
                                     sign='all',
                                     show_colorbar=True,
                                     title=title,
                                     plt_fig_axis=(fig, ax),
                                     cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data's label: {label}, {img_name} for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        """
        method to plot the attribution plots.

        Returns
        -------
        figs of the attribution plots, names of the attribution plots (for saving the plots with this name).

        """
        names = []
        figs = []
        max_images_per_plot = 4  # Define a constant for the maximum number of images per plot

        if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
            if self.learner.learner_config["layer"] == 'linear': 
                # Load the model
                cgan = load_cgan(layer='linear')
            elif self.learner.learner_config["layer"] == 'nlrl':
                # Load the model
                cgan = load_cgan(layer='nlrl')
            else:
                raise ValueError("Invalid values, it's either linear or nlrl")
                
            model = cgan.discriminator
            labels_list = self.learner.data_storage.get_item("labels")
        else:
            if self.learner.learner_config["layer"] == 'linear': 
                # Load the model
                gan = load_gan(layer='linear')
            elif self.learner.learner_config["layer"] == 'nlrl':
                # Load the model
                gan = load_gan(layer='nlrl')
            else:
                raise ValueError("Invalid values, it's either linear or nlrl")

            model = gan.discriminator
        
        # Custom cmap for better visulaization
        cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])

        fake_images = self.learner.data_storage.get_item("fake_images")
        real_images = self.learner.data_storage.get_item("real_images")
        
        class_dict = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
                          6: "6", 7: "7", 8: "8", 9: "9"}

        for types in ["real", "fake"]:
            inputs = real_images[-1].clone() if types == 'real' else fake_images[-1].clone()
    
            inputs.requires_grad = True  # Requires gradients set true
            
            # Get attribution maps for different techniques
            if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                labels = labels_list[-1]
                preds = model(inputs, labels)
                saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps_discriminator(model, inputs, labels)
            else:
                preds = model(inputs)
                saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps_discriminator(model, inputs)

            attrs = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution", "Occlusion"]

            # Process all input images
            total_indices = list(range(inputs.shape[0]))
            chunks = [total_indices[x:x + max_images_per_plot] for x in range(0, len(total_indices), max_images_per_plot)]

            for chunk in chunks:
                num_rows = len(chunk)
                num_cols = len(attrs) + 1  # One column for the original image, one for each attribution method
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
            
                # Adjust the shape of axs array if needed
                if num_rows == 1 and num_cols == 1:
                    axs = np.array([[axs]])
                elif num_rows == 1:
                    axs = axs[np.newaxis, :]
                elif num_cols == 1:
                    axs = axs[:, np.newaxis]
            
                count = 0
                for idx in chunk:
                    img = np.squeeze(inputs[idx].cpu().detach().numpy())  # Squeeze the image to 2D
                    pred = preds[idx]
                    
                    if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                        label = labels[idx].cpu().detach()
            
                    # Retrieve the attribution maps for the current image
                    results = [
                        np.squeeze(saliency_maps[idx].cpu().detach().numpy()),
                        np.squeeze(guided_backprop_maps[idx].cpu().detach().numpy()),
                        np.squeeze(input_x_gradient_maps[idx].cpu().detach().numpy()),
                        np.squeeze(deconv_maps[idx].cpu().detach().numpy()),
                        np.squeeze(occlusion_maps[idx].cpu().detach().numpy())
                    ]
            
                    # Display the original image
                    axs[count, 0].imshow(img, cmap='gray')             
                    if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                        axs[count, 0].set_title(f"Label: {label.item()}, Prediction: {pred.item():.3f}")
                    else:
                        axs[count, 0].set_title(f"Prediction: {pred.item():.3f}")
                    axs[count, 0].axis("off")
            
                    # Display each of the attribution maps next to the original image
                    for col, (attr, res) in enumerate(zip(attrs, results)):
                        title = f"{attr}"
                        
                        if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
                            self.safe_visualize(res, img, title, fig, axs[count, col + 1], label, class_dict[label.item()], types, cmap)
                        else:
                            # Call the visualization function, passing None for label and img_name since they are not applicable
                            self.safe_visualize(res, img, title, fig, axs[count, col + 1], None, None, types, cmap)
            
                    count += 1
            
                # Set the overall title for the figure based on the type of data
                fig.suptitle(f"Discriminator's view on {types.capitalize()} Data and the respective Attribution maps")
            
                # Store the figure with an appropriate name
                figs.append(fig)
                names.append(os.path.join("analysis_plots", "attribution_plots_discriminator", f"{types}_data", f"chunks_{chunks.index(chunk) + 1}"))
        return figs, names


class Hist_plot_discriminator(GenericPlot):
    def __init__(self, learner):
        super(Hist_plot_discriminator, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram of nlrl_ao plot")

    def consistency_check(self):
        return True
    
    def extract_parameters(self, model):
        for layer in model.modules():
            if isinstance(layer, NLRL_AO):
                negation = torch.sigmoid(layer.negation).detach().cpu().numpy()
                relevancy = torch.sigmoid(layer.relevancy).detach().cpu().numpy()
                selection = torch.sigmoid(layer.selection).detach().cpu().numpy()
    
                negation_init = torch.sigmoid(layer.negation_init).detach().cpu().numpy()
                relevancy_init = torch.sigmoid(layer.relevancy_init).detach().cpu().numpy()
                selection_init = torch.sigmoid(layer.selection_init).detach().cpu().numpy()
                
                return (negation, relevancy, selection), (negation_init, relevancy_init, selection_init)
        return None


    def plot(self):
        figs=[]
        names=[]
        
        labels = ['negation', 'relevancy', 'selection']
        
        if self.learner.learner_config["en_de_model"] == 'conditional_en_de':
            cgan = load_cgan(layer='nlrl')               
            model = cgan.discriminator
        else:
            gan = load_gan(layer='nlrl')
            model = gan.discriminator
            
        params, init_params = self.extract_parameters(model)
    
        for i, (param, init_param) in enumerate(zip(params, init_params)):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(init_param.ravel(), color='blue', alpha=0.5, bins=np.linspace(0, 1, 30), label='Initial')
            ax.hist(param.ravel(), color='red', alpha=0.5, bins=np.linspace(0, 1, 30), label='Trained')
            
            ax.set_title(f'{labels[i]} parameters distribution')
            ax.set_xlabel('sigmoid of the learnable weight matrices')
            ax.set_ylabel('number of parameters')
            ax.legend(loc='upper right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            figs.append(fig)
            names.append(os.path.join("histogram_plots", "discriminator", f"{labels[i]}"))
            
        return figs, names


class Hist_plot_classifier(GenericPlot):
    def __init__(self, learner):
        super(Hist_plot_classifier, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram of nlrl_ao plot")

    def consistency_check(self):
        return True
    
    def extract_parameters(self, model):
        for layer in model.modules():
            if isinstance(layer, NLRL_AO):
                negation = torch.sigmoid(layer.negation).detach().cpu().numpy()
                relevancy = torch.sigmoid(layer.relevancy).detach().cpu().numpy()
                selection = torch.sigmoid(layer.selection).detach().cpu().numpy()
    
                negation_init = torch.sigmoid(layer.negation_init).detach().cpu().numpy()
                relevancy_init = torch.sigmoid(layer.relevancy_init).detach().cpu().numpy()
                selection_init = torch.sigmoid(layer.selection_init).detach().cpu().numpy()
                
                return (negation, relevancy, selection), (negation_init, relevancy_init, selection_init)
        return None


    def plot(self):
        figs=[]
        names=[]
        
        labels = ['negation', 'relevancy', 'selection']
        
        model = load_classifier(layer='nlrl')
        params, init_params = self.extract_parameters(model)
    
        for i, (param, init_param) in enumerate(zip(params, init_params)):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(init_param.ravel(), color='blue', alpha=0.5, bins=np.linspace(0, 1, 30), label='Initial')
            ax.hist(param.ravel(), color='red', alpha=0.5, bins=np.linspace(0, 1, 30), label='Trained')
            
            ax.set_title(f'{labels[i]} parameters distribution')
            ax.set_xlabel('sigmoid of the learnable weight matrices')
            ax.set_ylabel('number of parameters')
            ax.legend(loc='upper right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            figs.append(fig)
            names.append(os.path.join("histogram_plots", "classifier", f"{labels[i]}"))
            
        return figs, names
    