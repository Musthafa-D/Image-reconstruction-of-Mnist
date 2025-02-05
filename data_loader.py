from ccbdl.data.utils.get_loader import get_loader
from torchvision import transforms
import matplotlib.pyplot as plt

def prepare_data(data_config):
    # augmentations_list = get_augmentation(data_config['augmentation'])
    # final_transforms = transforms.Compose(augmentations_list)

    # data_config["transform_input"] = final_transforms

    loader = get_loader(data_config["dataset"])
    train_data, test_data, val_data = loader(**data_config).get_dataloader()
    
    view_data(train_data, data_config)
    view_data(test_data, data_config)

    return train_data, test_data, val_data

def view_data(data, data_config):
    # View the first image in train_data or test_data
    batch = next(iter(data))
    inputs, labels = batch

    # Set up the subplot dimensions
    fig, axs = plt.subplots(2, 5, figsize=(15, 7))
    axs = axs.ravel()

    for idx in range(10):
        image = inputs[idx]
        label = labels[idx].item()  # Convert the label tensor to an integer

        image_np = image.permute(1, 2, 0).numpy()

        class_dict = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
                      6: "6", 7: "7", 8: "8", 9: "9"}

        # Display the image along with its label in the subplot
        axs[idx].imshow(image_np, cmap='gray')
        axs[idx].set_title(f"{class_dict[label]}")
        axs[idx].axis('off')

    plt.tight_layout()
    plt.show()

def get_augmentation(augmentations):
    transform_list = []
    for item in augmentations:
        if isinstance(item, str):  # Direct transform like RandomHorizontalFlip
            transform = getattr(transforms, item)()
            transform_list.append(transform)
        elif isinstance(item, dict):  # Transform with parameters like RandomRotation and Resize
            for name, params in item.items():
                if isinstance(params, list):  # If parameters are given as a list
                    transform = getattr(transforms, name)(*params)
                else:  # If a single parameter is given
                    transform = getattr(transforms, name)(params)
                transform_list.append(transform)
    return transform_list
