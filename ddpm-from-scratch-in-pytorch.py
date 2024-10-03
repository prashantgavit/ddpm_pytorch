
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.ddpm import DiffusionReverseProcess
from util.ddpm import train
from util.dataSetLoader import CustomMnistDataset






class CONFIG:
    model_path = 'ddpm_unet.pth'
    train_csv_path = 'data/digit-recognizer/train.csv'
    test_csv_path = 'data/digit-recognizer/test.csv'
    generated_csv_path = 'mnist_generated_data.csv'
    num_epochs = 50
    lr = 1e-4
    num_timesteps = 1000
    batch_size = 128
    img_size = 28
    in_channels = 1
    num_img_to_generate = 256



cfg = CONFIG()

train(cfg)

def generate(cfg):
    """
    Given Pretrained DDPM U-net model, Generate Real-life
    Images from noise by going backward step by step. i.e.,
    Mapping of Random Noise to Real-life images.
    """
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f'Device: {device}\n')
    
    # Initialize Diffusion Reverse Process
    drp = DiffusionReverseProcess()
    
    # Set model to eval mode
    model = torch.load(cfg.model_path).to(device)
    model.eval()
    
    # Generate Noise sample from N(0, 1)
    xt = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size).to(device)
    
    # Denoise step by step by going backward.
    with torch.no_grad():
        for t in reversed(range(cfg.num_timesteps)):
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(0).to(device))
            xt, x0 = drp.sample_prev_timestep(xt, noise_pred, torch.as_tensor(t).to(device))

    # Convert the image to proper scale
    xt = torch.clamp(xt, -1., 1.).detach().cpu()
    xt = (xt + 1) / 2
    
    return xt

cfg = CONFIG()


generated_imgs = []
for i in tqdm(range(cfg.num_img_to_generate)):
    xt = generate(cfg)
    xt = 255 * xt[0][0].numpy()
    generated_imgs.append(xt.astype(np.uint8).flatten())

generated_df = pd.DataFrame(generated_imgs, columns=[f'pixel{i}' for i in range(784)])
generated_df.to_csv(cfg.generated_csv_path, index=False)

from matplotlib import pyplot as plt
fig, axes = plt.subplots(8, 8, figsize=(5, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(np.reshape(generated_imgs[i], (28, 28)), cmap='gray')  # You might need to adjust the colormap based on your images
    ax.axis('off')  # Turn off axis labels

plt.tight_layout()  # Adjust spacing between subplots
plt.show()


def get_activation(dataloader, 
                   model, 
                   preprocess, # Preprocessing Transform for InceptionV3
                   device = 'cpu'
                  ):
    """
    Given Dataloader and Model, Generate N X 2048
    Dimensional activation map for N data points
    in dataloader.
    """
    
    # Set model to evaluation Mode
    model.to(device)
    model.eval()
    
    # Save activations
    pred_arr = np.zeros((len(dataloader.dataset), 2048))
    
    # Batch Size
    batch_size = dataloader.batch_size
    
    # Loop over Dataloader
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            
            # Transform the Batch according to Inceptionv3 specification
            batch = torch.stack([preprocess(img) for img in batch]).to(device)
            
            # Predict
            pred = model(batch).cpu().numpy()
            
            # Store
            pred_arr[i*batch_size : i*batch_size + batch.size(0), :] = pred
            
    return pred_arr


def calculate_activation_statistics(dataloader, 
                                    model, 
                                    preprocess, 
                                    device='cpu'
                                   ):
    """
    Get mean vector and covariance matrix of the activation maps.
    """
    
    # Get activation maps
    act = get_activation(dataloader, 
                         model, 
                         preprocess, # Preprocessing Transform for InceptionV3
                         device
                       )
    # Mean
    mu = np.mean(act, axis=0)
    
    # Covariance Metric
    sigma = np.cov(act, rowvar=False)
    
    return mu, sigma

from scipy import linalg

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    
    """
    Given Mean and Sigma of Real and Generated Data,
    it calculates FID between them using:
     
     d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
     
    """
    # Make sure they have appropriate dims
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle various cases
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean




import torchvision.transforms as transforms

transform_inception = transforms.Compose([
    transforms.Lambda(lambda x: (x + 1.0)/2.0), # [-1, 1] => [0, 1]
    transforms.ToPILImage(), # Tensor to PIL Image 
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB format
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization

])

import torchvision.models as models
from torchvision.models.inception import Inception_V3_Weights
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
model.fc = nn.Identity()

mnist_ds = CustomMnistDataset(cfg.generated_csv_path, cfg.num_img_to_generate)
mnist_dl = DataLoader(mnist_ds, cfg.batch_size//4, shuffle=False)
mu1, sigma1 = calculate_activation_statistics(mnist_dl, model, preprocess = transform_inception, device='cuda')

mnist_ds = CustomMnistDataset(cfg.test_csv_path, cfg.num_img_to_generate)
mnist_dl = DataLoader(mnist_ds, cfg.batch_size//4, shuffle=False)
mu2, sigma2 = calculate_activation_statistics(mnist_dl, model, preprocess = transform_inception, device='cuda')

# Calculate FID
fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
print(f'FID-Score: {fid}')

