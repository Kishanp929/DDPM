import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import numpy as np

class NoiseScheduler():
    def __init__(self, num_timesteps=50, type="linear", **kwargs):
        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        elif type == "cosine":
            self.init_cosine_schedule(**kwargs)  # Only pass relevant kwargs (like `s`)
        elif type == "sigmoid":
            self.init_sigmoid_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented")


    def init_linear_schedule(self, beta_start, beta_end):
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        self.alphas = 1 - self.betas
        self.alphas_cum_prods = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cum_prod = torch.sqrt(self.alphas_cum_prods)
        self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1 - self.alphas_cum_prods)

    # Indent this method inside the class
    #NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)
    def init_cosine_schedule(self, s):
        timesteps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1, dtype=torch.float32)

        alpha_cumprod = torch.cos((timesteps / self.num_timesteps) * (torch.pi / 2))**2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]  # Normalize

        self.alphas_cum_prods = alpha_cumprod[:-1]  # Reduce size to match num_timesteps
        self.betas = 1 - (self.alphas_cum_prods[1:] / self.alphas_cum_prods[:-1])

        # Ensure betas has same size as alphas_cum_prods (T)
        self.betas = torch.cat([self.betas, self.betas[-1:]])  # Duplicate last value

        self.alphas = 1 - self.betas
        self.sqrt_alphas_cum_prod = torch.sqrt(self.alphas_cum_prods)
        self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1 - self.alphas_cum_prods)
    def init_sigmoid_schedule(self, slope):

        timesteps = torch.linspace(0, 3, self.num_timesteps, dtype=torch.float32)  # (200,)
        sigmoid_curve = torch.sigmoid(slope * timesteps)  # (200,)
        alpha_cumprod = 1 - sigmoid_curve  # (200,)

        self.alphas_cum_prods = alpha_cumprod  # (200,)

        # Compute betas (ensure 200 elements)
        self.betas = 1 - (self.alphas_cum_prods[1:] / self.alphas_cum_prods[:-1])  # (199,)
        self.betas = torch.cat([self.betas, self.betas[-1:]])  # Now (200,)

        self.alphas = 1 - self.betas  # (200,)
        self.sqrt_alphas_cum_prod = torch.sqrt(self.alphas_cum_prods)  # (200,)
        self.sqrt_one_minus_alphas_cum_prod = torch.sqrt(1 - self.alphas_cum_prods)  # (200,)

    def __len__(self):
        return self.num_timesteps




class DDPM(nn.Module):

    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        """
        super(DDPM, self).__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps

        # A simple time embedding network. Here we embed the timestep into a 128-dim vector.
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # The main model predicts noise given the noisy input and the time embedding.
        self.model = nn.Sequential(
            nn.Linear(n_dim + 128, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, n_dim)
        )

    def forward(self, x, t):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        # Ensure t has shape [batch_size, 1]
        t = t.unsqueeze(-1).float()
        t_emb = self.time_embed(t)
        # Concatenate the time embedding with the input
        x_in = torch.cat([x, t_emb], dim=1)
        return self.model(x_in)




class ConditionalDDPM(nn.Module):
    def __init__(self, n_classes=2, n_dim=3, n_steps=200, hidden_dim=128):
        """
        Conditional DDPM noise prediction model.

        Args:
            n_classes: int, number of classes in the dataset.
            n_dim: int, input data dimensionality.
            n_steps: int, number of steps in the diffusion process.
            hidden_dim: int, size of hidden layers.
        """
        super(ConditionalDDPM, self).__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.n_classes = n_classes

        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Class embedding
        self.class_embed = nn.Embedding(n_classes, 128)

        # The main model predicts noise given the noisy input, time embedding, and class embedding.
        self.model = nn.Sequential(
            nn.Linear(512, 256),  # Concatenated input size
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, n_dim)
        )

    def forward(self, x, t, y):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim].
            t: torch.Tensor, the timestep tensor [batch_size].
            y: torch.Tensor, the class labels [batch_size] (for conditional generation).

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim].
        """
        # Ensure t has shape [batch_size, 1]
        
        t = t.unsqueeze(-1).float()  # Shape [batch_size, 1]
        t = t.to(next(self.time_embed.parameters()).device)
        y = y.to(next(self.class_embed.parameters()).device)
        t_emb = self.time_embed(t)  # Shape [batch_size, hidden_dim]
        
        batch_size = x.shape[0]
        # Embed class label
        y_emb = self.class_embed(y)  # Shape [batch_size, hidden_dim]

        t_emb = t_emb.view(t_emb.shape[0], -1)
        y_emb = y_emb.view(y_emb.shape[0], -1)
      
        # Flatten x if using an MLP
        x = x.view(batch_size, -1).to(device)  # Flatten spatial dimensions
        
        x_in = torch.cat([x, t_emb, y_emb], dim=1)  # Ensure all have [batch_size, feature_dim]
        pad_size = 512 - x_in.shape[1]
        if pad_size > 0:
            x_in = torch.cat([x_in, torch.zeros(x_in.shape[0], pad_size, dtype=x_in.dtype, device=x_in.device)], dim=1)

       
        return self.model(x_in)


class ClassifierDDPM:
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model.
    """

    def __init__(self, model: ConditionalDDPM, noise_scheduler):
        """
        Args:
            model: ConditionalDDPM, the trained conditional diffusion model.
            noise_scheduler: NoiseScheduler, handles diffusion process.
        """
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.n_classes = model.n_classes

    def __call__(self, x):
        return self.predict(x)

    def predict_proba(self, x):
        """
        Computes class probabilities using the DDPM model.

        Args:
            x: torch.Tensor, input data tensor [batch_size, n_dim]
        
        Returns:
            torch.Tensor, probabilities for each class [batch_size, n_classes]
        """
        batch_size = x.shape[0]
        device = x.device

        # Compute probability for each class
        probs = []
        for y in range(self.n_classes):
            y_tensor = torch.full((batch_size,), y, dtype=torch.long, device=device)
            t = torch.randint(0, self.noise_scheduler.n_steps, (batch_size,), device=device)

            # Predict noise
            noise_pred = self.model(x, t, y_tensor)
            
            # Compute likelihood (lower noise → higher probability)
            likelihood = -torch.norm(noise_pred, dim=1)  # Negative L2 norm
            probs.append(likelihood)

        # Convert to probability distribution
        probs = torch.stack(probs, dim=1)  # [batch_size, n_classes]
        probs = F.softmax(probs, dim=1)  # Normalize to probability

        return probs

    def predict(self, x):
        """
        Predicts the class with the highest probability.

        Args:
            x: torch.Tensor, input data tensor [batch_size, n_dim]
        
        Returns:
            torch.Tensor, predicted class indices [batch_size]
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)  # Return class with highest probability

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    device = next(model.parameters()).device
    model.train()

    if(isinstance(model, DDPM)):
        for epoch in range(epochs):
            losses = []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for x in pbar:
                x = x.to(device)
                optimizer.zero_grad()
                batch_size = x.shape[0]

                # Sample a random timestep for each sample in the batch
                t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device)
                # Get corresponding coefficients for the sampled timesteps
                sqrt_alphas = noise_scheduler.sqrt_alphas_cum_prod.to(device)[t]
                sqrt_one_minus = noise_scheduler.sqrt_one_minus_alphas_cum_prod.to(device)[t]

                # Sample noise and create a noisy version of x
                noise = torch.randn_like(x)
                noisy_x = sqrt_alphas.unsqueeze(-1) * x + sqrt_one_minus.unsqueeze(-1) * noise

                # Predict the noise using the model
                pred_noise = model(noisy_x, t)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                pbar.set_description(f"Epoch {epoch+1} Loss {np.mean(losses):.4f}")
            # Save model checkpoint for each epoch if desired
            torch.save(model.state_dict(), os.path.join(run_name, f"model_epoch{epoch+1}.pth"))
        # Save final model
        torch.save(model.state_dict(), os.path.join(run_name, "model.pth"))

    
    else:
        for epoch in range(epochs):
            losses = []
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            for batch in pbar:
                x, y = batch  # Expect (x, y) for ConditionalDDPM
                y = y.to(device)
                

                x = x.to(device)
                optimizer.zero_grad()
                batch_size = x.shape[0]

                # Sample random timesteps
                t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device)

                # Get noise coefficients
                sqrt_alphas = noise_scheduler.sqrt_alphas_cum_prod.to(device)[t].view(batch_size, 1, 1, 1)
                sqrt_one_minus = noise_scheduler.sqrt_one_minus_alphas_cum_prod.to(device)[t].view(batch_size, 1, 1, 1)

                # Sample noise and create noisy data
                noise = torch.randn_like(x)
                noisy_x = sqrt_alphas * x + sqrt_one_minus * noise

                # Predict the noise using the model (Conditionally or Unconditionally)
                
                pred_noise = model(noisy_x, t, y)  # Conditional Model
                

                # Compute loss and update model
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                pbar.set_description(f"Epoch {epoch+1} Loss {np.mean(losses):.4f}")

            # Save model checkpoint for each epoch
            torch.save(model.state_dict(), os.path.join(run_name, f"model_epoch{epoch+1}.pth"))

        # Save final model
        torch.save(model.state_dict(), os.path.join(run_name, "model.pth"))

    

@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False, init_noise=None):
    """
    Sample from the model

    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        return_intermediate: bool
        init_noise: Optional initial noise tensor of shape [n_samples, n_dim]. If provided, uses this instead of standard normal.
    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
        or list of tensors for each timestep if return_intermediate=True
    """
    device = next(model.parameters()).device
    T = noise_scheduler.num_timesteps

    # Load initial noise from the given file if not provided
    if init_noise is not None:
        x = init_noise.to(device)
    else:
        x = torch.randn(n_samples, model.n_dim, device=device)
    x = x[:n_samples]
    intermediates = [x.clone()] if return_intermediate else None

    # Reverse diffusion process
    for t in reversed(range(1,T)):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor)
        beta_t = noise_scheduler.betas[t].to(device)
        alpha_t = noise_scheduler.alphas[t].to(device)
        alpha_cum = noise_scheduler.alphas_cum_prods[t].to(device)
        sqrt_one_minus = noise_scheduler.sqrt_one_minus_alphas_cum_prod[t].to(device)

        # Compute the mean (reverse process) as in [Ho et al., 2020]
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / sqrt_one_minus
        x = coef1 * (x - coef2 * pred_noise)

        # Add noise except for the final step
        if t > 1:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = x + sigma * noise

        if return_intermediate:
            intermediates.append(x.clone())

    return intermediates if return_intermediate else x



def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Classifier-Free Guidance (CFG) Sampling Function.

    Args:
        model: The trained ConditionalDDPM model.
        n_samples: Number of samples to generate.
        noise_scheduler: The NoiseScheduler object.
        guidance_scale: Scale factor for classifier-free guidance.
        class_label: The class label for conditional generation.
        device: Device to run sampling on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Generated samples.
    """
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device
    # Initialize pure Gaussian noise
    x = torch.randn(n_samples, model.n_dim, dtype=torch.float32).to(device)
    y = torch.full((n_samples,), class_label, dtype=torch.long).to(device)
    y_uncond = torch.full((n_samples,), 0, dtype=torch.long).to(device)

    for t in reversed(range(1,noise_scheduler.num_timesteps)):
        
  # Ensure correct dtype
        t_tensor = torch.full((n_samples, 1), t, dtype=torch.long).to(device)  # Ensure integer type
        
        t = t_tensor[0, 0].item()  # Convert to scalar integer for indexing


   
        
   
        #torch.Size([128, 1, 128, 2]) torch.Size([128]) torch.Size([128])
        # Predict noise for both conditional and unconditional cases
        noise_pred_cond = model(x, t_tensor, y)  # Conditioned on y
        noise_pred_uncond = model(x, t_tensor, y_uncond)  # Unconditioned (random)
        # Classifier-Free Guidance: Weighted interpolation
        noise_pred = (1 + guidance_scale) * noise_pred_cond - guidance_scale * noise_pred_uncond
        
        # Reverse diffusion step
        
        
        
        alpha_t = noise_scheduler.alphas_cum_prods[t]  # Get αₜ
        beta_t = noise_scheduler.betas[t]  # Get βₜ
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1 - alpha_t, min=1e-5))
          
        x = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t  # Denoised update
        if t > 1:
            noise = torch.randn_like(x)  # Add stochastic noise
            x = x + torch.sqrt(beta_t) * noise  # Final step adjustment

    return x  # 






def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn): #bonus
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    # parser.add_argument("--n_steps", type=int, default=None)
    # parser.add_argument("--lbeta", type=float, default=None)
    # parser.add_argument("--ubeta", type=float, default=None)
    # parser.add_argument("--epochs", type=int, default=None)
    # parser.add_argument("--n_samples", type=int, default=None)
    # parser.add_argument("--batch_size", type=int, default=None)
    # parser.add_argument("--lr", type=float, default=None)
    # parser.add_argument("--dataset", type=str, default = None)
    # parser.add_argument("--seed", type=int, default = 42)
    # parser.add_argument("--n_dim", type=int, default = 3)

    parser.add_argument("--mode", choices=['train', 'sample'], default='train')
    parser.add_argument("--model", choices=['ConditionalDDPM', 'DDPM'], default='DDPM')
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--ns_type", choices=['linear', 'cosine', 'sigmoid'], default='linear')
                                                                       # Increase diffusion steps for smoother denoising
    parser.add_argument("--lbeta", type=float, default=1e-4) #c1
    parser.add_argument("--ubeta", type=float, default=0.03)# c2
    parser.add_argument("--s", type=float, default=0.08)
    parser.add_argument("--slope", type=float, default=5.00)        # Slightly lower ubeta for gentler noise increments
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument('--class_label', type=int, default=0, help="Class label for sampling")
    parser.add_argument("--epochs", type=int, default=150)             # Train for more epochs
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)              # Lower learning rate for smoother convergence
    parser.add_argument("--dataset", type=str, default="moons")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_dim", type=int, default=2)



    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}' # can include more hyperparams
    os.makedirs(run_name, exist_ok=True)
    if(args.model=="ConditionalDDPM"):
        model = ConditionalDDPM(n_classes=2, n_dim=args.n_dim, n_steps=args.n_steps)
    elif(args.model=="DDPM"):
        model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    if(args.ns_type=='sigmoid'):
        noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps , type = args.ns_type ,slope=args.slope)
    elif(args.ns_type=='cosine'):
        noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps , type = args.ns_type ,s=args.s)
    else:
        noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, type = args.ns_type , beta_start=args.lbeta, beta_end=args.ubeta)
    model = model.to(   device   )

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        # can split the data into train and test -- for evaluation later
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        if(isinstance(model, DDPM)):
            dataloader = torch.utils.data.DataLoader(data_X, batch_size=args.batch_size, shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y), batch_size=args.batch_size, shuffle=True, drop_last=True)

        train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth', map_location=device))
        if(isinstance(model, DDPM)):
            samples = sample(model, args.n_samples, noise_scheduler)
        else:
            
            samples = sampleCFG(model, args.n_samples, noise_scheduler, args.guidance_scale, args.class_label)
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
        # nll_value = utils.get_nll(data_X, samples, temperature=1e-1)
        # print("Negative Log Likelihood:", nll_value.item())

        # # Compute Likelihood
        # likelihood = utils.get_likelihood(data_X, samples, temperature=1e-1)
        # print("Likelihood:", likelihood.item())

        # # Compute Gaussian Kernel
        # kernel_value = utils.gaussian_kernel(samples[0], data_X, temperature=1e-1)
        # print("Gaussian Kernel Value:", kernel_value.item())


        # # emd_value = utils.get_emd(data_X_np, samples_np)
        # # print("Earth Mover’s Distance (EMD):", emd_value)

    else:
        raise ValueError(f"Invalid mode {args.mode}")