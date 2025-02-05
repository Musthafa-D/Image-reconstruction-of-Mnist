import torch
import os
from ccbdl.utils import DEVICE
from Networks.classfiers import CNN
from Networks.discriminators import GAN, CGAN
from ccbdl.config_loader.loaders import ConfigurationLoader
from captum.attr import Saliency, GuidedBackprop, InputXGradient, Deconvolution, Occlusion
from torch.utils.data import Dataset


def generate_images(ddpm, device, noisy_images, denoising_option, labels=None):
    """
    Given a DDPM model, a number of samples (like batch size)
    to be generated and a device, returns some newly generated samples

    Parameters
    ----------
    ddpm : Denoising Diffusion Model.
    
    device : CPU or GPU.
    
    noisy_images : torch.FloatTensor or torch.cuda.FloatTensor based on device (optional), 
    Noisy_images to be denoised.
    
    labels : torch.LongTensor or torch.cuda.LongTensor based on device (required for cddpm),
    labels of the respective images. 
    The default is None.

    Returns
    -------
    x : generated images after denoising the noises.
    noise: noise values that were used to generate images.

    """
    with torch.no_grad():
        x = noisy_images
        n_samples = x.size(0)
        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            
            if labels is not None:
                noise_pred = ddpm.noise_prediction_labels(x, time_tensor, labels)             
            else:
                noise_pred = ddpm.noise_prediction(x, time_tensor)
                
            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * noise_pred)

            if t > 0:
                # Add Noise (Langevin Dynamics Fashion)
                z = torch.randn_like(noise_pred).to(device)

                beta_t = ddpm.betas[t]          
                if denoising_option == 1:
                    # Option 1: sigma_t squared = beta_t
                    sigma_t = beta_t.sqrt()
                
                elif denoising_option == 2:
                    # Option 2: sigma_t squared = beta_tilda_t
                    prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                    beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                    sigma_t = beta_tilda_t.sqrt()
                
                else:
                    raise ValueError("Invalid value for model: denoising_option, it should be 1, or 2.")

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z
    return x

def generate_images_latent(ddpm, device, noisy_images_latent, denoising_option, labels=None):
    """
    Given a DDPM model, a number of samples (like batch size)
    to be generated and a device, returns some newly generated samples

    Parameters
    ----------
    ddpm : Denoising Diffusion Model.
    
    device : CPU or GPU.
    
    noisy_images : torch.FloatTensor or torch.cuda.FloatTensor based on device (optional), 
    Noisy_images to be denoised.
    
    labels : torch.LongTensor or torch.cuda.LongTensor based on device (required for cddpm),
    labels of the respective images. 
    The default is None.

    Returns
    -------
    x : generated images after denoising the noises.
    noise: noise values that were used to generate images.

    """
    with torch.no_grad():
        x = noisy_images_latent
        n_samples = x.size(0)
        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            
            if labels is not None:
                noise_pred = ddpm.noise_prediction_labels(x, time_tensor, labels)             
            else:
                noise_pred = ddpm.noise_prediction(x, time_tensor)
                
            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * noise_pred)

            if t > 0:
                # Add Noise (Langevin Dynamics Fashion)
                z = torch.randn_like(noise_pred).to(device)

                beta_t = ddpm.betas[t]          
                if denoising_option == 1:
                    # Option 1: sigma_t squared = beta_t
                    sigma_t = beta_t.sqrt()
                
                elif denoising_option == 2:
                    # Option 2: sigma_t squared = beta_tilda_t
                    prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                    beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                    sigma_t = beta_tilda_t.sqrt()
                
                else:
                    raise ValueError("Invalid value for model: denoising_option, it should be 1, or 2.")

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z
        x_de = ddpm.decoder(x)
    return x, x_de

def generate_new_images(ddpm, n_samples, device, channels, height, width, denoising_option, fixed_noise=None, labels=None):
    """
    Given a DDPM model, a number of samples (like batch size)
    to be generated and a device, returns some newly generated samples

    Parameters
    ----------
    ddpm : Denoising Diffusion Model.
    
    n_samples : int, 
    Number of samples to be generated.
    
    device : CPU or GPU.
    
    channels : int, 
    1 for Greyscale and 3 for RGB.
    
    height : int, 
    Height of each image data.
    
    width : int, 
    Width of each image data.
    
    fixed_noise : torch.FloatTensor or torch.cuda.FloatTensor based on device (optional), 
    Noise value to be denoised for image generation.
    The default is None.
    
    labels : torch.LongTensor or torch.cuda.LongTensor based on device (required for cddpm),
    labels of the respective images. 
    The default is None.

    Returns
    -------
    x : generated images after denoising the noises.
    noise: noise values that were used to generate images.

    """
    with torch.no_grad():
        if fixed_noise is None:
            noise = torch.randn(n_samples, channels, height, width).to(device)
            x = noise
        else:
            # Use the provided fixed noise
            noise = fixed_noise
            x = noise

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            if labels is not None:
                noise_pred = ddpm.noise_prediction_labels(x, time_tensor, labels)             
            else:
                noise_pred = ddpm.noise_prediction(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * noise_pred)

            if t > 0:
                # Add Noise (Langevin Dynamics Fashion)
                z = torch.randn(n_samples, channels, height, width).to(device)

                beta_t = ddpm.betas[t]
                if denoising_option == 1:
                    # Option 1: sigma_t squared = beta_t
                    sigma_t = beta_t.sqrt()
                
                elif denoising_option == 2:
                    # Option 2: sigma_t squared = beta_tilda_t
                    prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                    beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                    sigma_t = beta_tilda_t.sqrt()
                
                else:
                    raise ValueError("Invalid value for model: denoising_option, it should be 1, or 2.")

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z
    return x, noise

def generate_new_images_latent(ddpm, n_samples, device, channels, height, width, denoising_option, fixed_noise=None, labels=None):
    """
    Given a DDPM model, a number of samples (like batch size)
    to be generated and a device, returns some newly generated samples

    Parameters
    ----------
    ddpm : Denoising Diffusion Model.
    
    n_samples : int, 
    Number of samples to be generated.
    
    device : CPU or GPU.
    
    channels : int, 
    1 for Greyscale and 3 for RGB.
    
    height : int, 
    Height of each image data.
    
    width : int, 
    Width of each image data.
    
    fixed_noise : torch.FloatTensor or torch.cuda.FloatTensor based on device (optional), 
    Noise value to be denoised for image generation.
    The default is None.
    
    labels : torch.LongTensor or torch.cuda.LongTensor based on device (required for cddpm),
    labels of the respective images. 
    The default is None.

    Returns
    -------
    x : generated images after denoising the noises.
    noise: noise values that were used to generate images.

    """
    with torch.no_grad():
        if fixed_noise is None:
            noise = torch.randn(n_samples, channels, height, width).to(device)
            x = noise
        else:
            # Use the provided fixed noise
            noise = fixed_noise
            x = noise

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            if labels is not None:
                noise_pred = ddpm.noise_prediction_labels(x, time_tensor, labels)             
            else:
                noise_pred = ddpm.noise_prediction(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * noise_pred)

            if t > 0:
                # Add Noise (Langevin Dynamics Fashion)
                z = torch.randn(n_samples, channels, height, width).to(device)

                beta_t = ddpm.betas[t]
                if denoising_option == 1:
                    # Option 1: sigma_t squared = beta_t
                    sigma_t = beta_t.sqrt()
                
                elif denoising_option == 2:
                    # Option 2: sigma_t squared = beta_tilda_t
                    prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                    beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                    sigma_t = beta_tilda_t.sqrt()
                
                else:
                    raise ValueError("Invalid value for model: denoising_option, it should be 1, or 2.")

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z
        x_de = ddpm.decoder(x)
    return x, x_de, noise

def grayscale_to_rgb(imgs):
    """
    Images is expected to be of shape [batch_size, 1, height, width]

    Parameters
    ----------
    imgs

    Returns
    -------
    Images with 3 channels i.e. as RGB images

    """
    return imgs.repeat(1, 3, 1, 1)

def load_classifier_rgb(layer:str):
    """
    Loads the pre trained classifier
    
    Parameters
    ----------
    layer: 'linear' or 'nlrl'

    Returns
    -------
    trained classifier (trained on mnist rgb dataset)

    """
    config = ConfigurationLoader().read_config("config.yaml")
    
    if layer == 'linear':
        classifier_config = config["classifier_linear"]  
        classifier = CNN(3,"Classifier to classify real and fake images", 
                      **classifier_config).to(DEVICE)
        checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_rgb_linear.pt")
    else:
        classifier_config = config["classifier_nlrl"]
        classifier = CNN(3,"Classifier to classify real and fake images", 
                      **classifier_config).to(DEVICE)
        checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_rgb_nlrl.pt")
    
    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    return classifier

def load_classifier(layer:str):
    """
    Loads the pre trained classifier
    
    Parameters
    ----------
    layer: 'linear' or 'nlrl'

    Returns
    -------
    trained classifier (trained on mnist dataset)

    """
    config = ConfigurationLoader().read_config("config.yaml")
    
    if layer == 'linear':
        classifier_config = config["classifier_linear"]  
        classifier = CNN(1,"Classifier to classify real and fake images", 
                      **classifier_config).to(DEVICE)
        checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_linear.pt")
    else:
        classifier_config = config["classifier_nlrl"]
        classifier = CNN(1,"Classifier to classify real and fake images", 
                      **classifier_config).to(DEVICE)
        checkpoint_path = os.path.join("Saved_networks", "cnn_net_best_nlrl.pt")
    
    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    return classifier

def load_gan(layer:str):
    """
    Loads the pre trained gan
    
    Parameters
    ----------
    layer: 'linear' or 'nlrl'

    Returns
    -------
    trained gan (trained on mnist dataset)

    """
    config = ConfigurationLoader().read_config("config.yaml")
    
    if layer == 'linear':
        discriminator_config = config["discriminator_linear"]  
        gan = GAN(**discriminator_config).to(DEVICE)
        checkpoint_path = os.path.join("Saved_networks", "dis_net_best_linear.pt")
    else:
        discriminator_config = config["discriminator_nlrl"]  
        gan = GAN(**discriminator_config).to(DEVICE)
        checkpoint_path = os.path.join("Saved_networks", "dis_net_best_nlrl.pt")
    
    checkpoint = torch.load(checkpoint_path)
    gan.load_state_dict(checkpoint['model_state_dict'])
    gan.eval()
    return gan

def load_cgan(layer:str):
    """
    Loads the pre trained conditional gan
    
    Parameters
    ----------
    layer: 'linear' or 'nlrl'

    Returns
    -------
    trained conditional gan (trained on mnist dataset)

    """
    config = ConfigurationLoader().read_config("config.yaml")
    
    if layer == 'linear':
        discriminator_config = config["discriminator_linear"]  
        cgan = CGAN(**discriminator_config).to(DEVICE)
        checkpoint_path = os.path.join("Saved_networks", "dis_c_net_best_linear.pt")
    else:
        discriminator_config = config["discriminator_nlrl"]  
        cgan = CGAN(**discriminator_config).to(DEVICE)
        checkpoint_path = os.path.join("Saved_networks", "dis_c_net_best_nlrl.pt")
    
    checkpoint = torch.load(checkpoint_path)
    cgan.load_state_dict(checkpoint['model_state_dict'])
    cgan.eval()
    return cgan

def attributions(model):
    """
    Captum's attributions based on the given model

    Parameters
    ----------
    model : classifier or discriminator.

    Returns
    -------
    saliency, guided_backprop, input_x_gradient, deconv, and occlusion

    """
    # Initialize the Saliency object
    saliency = Saliency(model)
    # Initialize the GuidedBackprop object
    guided_backprop = GuidedBackprop(model)
    # Initialize the DeepLift object
    input_x_gradient = InputXGradient(model)
    # Initialize the Deconvolution object
    deconv = Deconvolution(model)
    # Initialize the Occlusion object
    occlusion = Occlusion(model)  
    return saliency, guided_backprop, input_x_gradient, deconv, occlusion

def attribution_maps_classifier(model, inputs, labels):
    """
    Captum's attribution maps based on the respective classifier, inputs and labels

    Parameters
    ----------
    model : classifier.
    
    inputs : images from the dataset.
    
    labels : labels of the respective images (here it is the predicted labels).

    Returns
    -------
    saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, and occlusion_maps.

    """
    saliency, guided_backprop, input_x_gradient, deconv, occlusion = attributions(model)
    
    saliency_maps = saliency.attribute(inputs, target=labels)
    guided_backprop_maps = guided_backprop.attribute(inputs, target=labels)
    input_x_gradient_maps = input_x_gradient.attribute(inputs, target=labels)
    deconv_maps = deconv.attribute(inputs, target=labels)
    occlusion_maps = occlusion.attribute(inputs, target=labels, sliding_window_shapes=(1, 3, 3), strides=(1, 2, 2))   
    return saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps

def attribution_maps_discriminator(model, inputs, labels=None):
    """
    Captum's attribution maps based on the respective classifier, inputs and labels

    Parameters
    ----------
    model : discriminator.
    
    inputs : images from the dataset or generated images.
    
    labels : labels of the respective images (required for cgan),
    The default is None.

    Returns
    -------
    saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, and occlusion_maps.

    """
    saliency, guided_backprop, input_x_gradient, deconv, occlusion = attributions(model)
    
    if labels is not None:
        saliency_maps = saliency.attribute(inputs, additional_forward_args=labels)
        guided_backprop_maps = guided_backprop.attribute(inputs, additional_forward_args=labels)
        input_x_gradient_maps = input_x_gradient.attribute(inputs, additional_forward_args=labels)
        deconv_maps = deconv.attribute(inputs, additional_forward_args=labels)
        occlusion_maps = occlusion.attribute(inputs, additional_forward_args=labels, sliding_window_shapes=(1, 3, 3), strides=(1, 2, 2))
    else:
        # For GAN, we do not pass the labels
        saliency_maps = saliency.attribute(inputs)
        guided_backprop_maps = guided_backprop.attribute(inputs)
        input_x_gradient_maps = input_x_gradient.attribute(inputs)
        deconv_maps = deconv.attribute(inputs)
        occlusion_maps = occlusion.attribute(inputs, sliding_window_shapes=(1, 3, 3), strides=(1, 2, 2))
    return saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps


class ImageTensorDataset(Dataset):
    """
    Custom Dataset class to handle lists of tensors (images)
    
    """
    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]


def sinusoidal_embedding(n_steps, time_emb_dim):
    # Returns the standard positional embedding
    embedding = torch.zeros(n_steps, time_emb_dim)
    wk = torch.tensor([1 / 10000 ** (2 * j / time_emb_dim) for j in range(time_emb_dim)])
    wk = wk.reshape((1, time_emb_dim))
    t = torch.arange(n_steps).reshape((n_steps, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding