import argparse
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision import utils, transforms
from PIL import Image

def gaussian_blur(img, kernel_size=61, sigma=3.0):
    """Apply Gaussian blur to a batch of images."""
    blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return blur(img)

def to_image(tensor):
    """Convert tensor to numpy array for saving."""
    return ((tensor.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

def main():
    args = create_argparser().parse_args()

    try:
        dist_util.setup_dist()
        logger.configure(dir=args.sample_dir)

        # Load and preprocess ground-truth image
        logger.log("loading ground-truth image...")
        if not os.path.exists(args.input_image):
            raise FileNotFoundError(f"Input image {args.input_image} not found")
        image = Image.open(args.input_image).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Explicitly resize to 256x256
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        x_gt = transform(image).unsqueeze(0).to(dist_util.dev())
        logger.log(f"Input image shape after transform: {x_gt.shape}")  # Should be [1, 3, 256, 256]

        # Create blurred observation
        logger.log("creating blurred observation...")
        y = gaussian_blur(x_gt, kernel_size=args.blur_kernel, sigma=args.blur_sigma)
        y = y + torch.randn_like(y) * args.noise_std  # Add noise
        # Save blurred image for reference
        if dist.get_rank() == 0:
            utils.save_image(y, os.path.join(logger.get_dir(), "blurred_input.png"), nrow=1, normalize=True)

        logger.log("creating model and diffusion...")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} not found")
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        # Verify model input size
        if args.image_size != 256:
            logger.log(f"Warning: Model expects image_size={args.image_size}, but input is resized to 256x256")

        logger.log("sampling with DPS...")
        all_images = []
        all_labels = []
        total_generated = 0

        while total_generated < args.num_samples:
            current_batch_size = min(args.batch_size, args.num_samples - total_generated)

            model_kwargs = {}
            if args.class_cond:
                classes = torch.randint(
                    low=0, high=NUM_CLASSES, size=(current_batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes

            # Initialize noisy image
            x_t = torch.randn((current_batch_size, 3, 256, 256), device=dist_util.dev())  # Hardcode 256x256
            # Repeat blurred observation for batch
            y_batch = y.repeat(current_batch_size, 1, 1, 1)

            # Custom DPS sampling loop
            for t in range(diffusion.num_timesteps - 1, -1, -1):  # Reverse loop from T-1 to 0
                timesteps = torch.full((current_batch_size,), t, device=dist_util.dev(), dtype=torch.long)
                x_t = x_t.detach().requires_grad_(True)

                # Predict noise
                with torch.no_grad():
                    noise_pred = model(x_t, timesteps, **model_kwargs)
                    # If model outputs 6 channels, take only the first 3 (mean prediction)
                    if noise_pred.shape[1] == 6:
                        noise_pred = noise_pred[:, :3, :, :]  # Select first 3 channels
                    logger.log(f"Step {t}, x_t shape: {x_t.shape}, noise_pred shape: {noise_pred.shape}")

                # Get scheduler parameters
                beta_t = diffusion.betas[t]
                alpha_t = 1.0 - beta_t
                alpha_bar_t = diffusion.alphas_cumprod[t]
                sigma_t = beta_t ** 0.5

                # Predict x0
                x0_pred = (x_t - (1 - alpha_bar_t) ** 0.5 * noise_pred) / alpha_bar_t ** 0.5

                # DPS gradient step
                blurred = gaussian_blur(x0_pred, kernel_size=args.blur_kernel, sigma=args.blur_sigma) + torch.randn_like(x0_pred) * args.noise_std
                #loss = F.mse_loss(blurred  , y_batch)
                residual = blurred - y_batch
                loss = 0.5 * ((blurred - y_batch) ** 2).sum()

                grad = torch.autograd.grad(loss, x_t)[0]
                grad_norm = grad.norm()

                # Adaptive step size based on loss norm (from intuition code)
                rho = 1.0 / (residual.norm().detach() )

                # Update x_t with gradient guidance
                x_t = x_t.detach() - rho * grad * (1 - alpha_bar_t)

                # Standard diffusion update (for DDPM)
                if t > 0:
                    noise = torch.randn_like(x_t)
                    x_t = (1 / alpha_t ** 0.5) * (x_t - ((1 - alpha_t) / (1 - alpha_bar_t) ** 0.5 * noise_pred)) + sigma_t * noise

                x_t = x_t.clamp(-2, 2)

                if t % 50 == 0 and dist.get_rank() == 0:
                    logger.log(f"Step {t}, Loss: {loss.item():.4f}, Rho: {rho.item():.4f}")

            # Save generated images
            for i in range(current_batch_size):
                out_path = os.path.join(logger.get_dir(), f"{str(total_generated + i).zfill(5)}.png")
                utils.save_image(
                    x0_pred[i].unsqueeze(0),
                    out_path,
                    nrow=1,
                    normalize=True,
                )

            # Prepare for NPZ saving
            x0_pred = to_image(x0_pred)
            gathered_samples = [torch.zeros_like(torch.tensor(x0_pred)) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, torch.tensor(x0_pred).to(dist_util.dev()))
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

            if args.class_cond:
                gathered_labels = [torch.zeros_like(classes) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

            total_generated += current_batch_size
            logger.log(f"created {total_generated} / {args.num_samples} samples")

        # Save NPZ
        if dist.get_rank() == 0:
            arr = np.concatenate(all_images, axis=0)
            arr = arr[: args.num_samples]
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                label_arr = np.concatenate(all_labels, axis=0)
                label_arr = label_arr[: args.num_samples]
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

        dist.barrier()
        logger.log("sampling complete")

    except Exception as e:
        logger.log(f"Error occurred: {str(e)}")
        raise
    finally:
        dist.destroy_process_group()

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="/content/P2-weighting/ffhq_baseline.pt",
        sample_dir="",
        input_image="/content/69165.png",
        blur_kernel=61,
        blur_sigma=3.0,
        noise_std=0.05,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
