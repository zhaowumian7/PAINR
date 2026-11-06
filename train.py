"""Train the single-speed photoacoustic neural representation."""

import datetime
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tinycudann as tcnn
import torch
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataset
import utils


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    output_root = project_root / "outputs"
    result_dir = output_root / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(result_dir / "tensorboard_logs")
    gc.collect()
    torch.cuda.empty_cache()

    sin_path = project_root / "data/raw/img1_128_Norm.nii"
    lr = 1e-3
    num_sensors = 128
    num_epochs = 80
    gpu = 0
    batch_size = 8
    sample_N = 1200
    radius = 0.04
    field_of_view = 0.04
    num_rays = 400
    speed_of_sound = 1500
    time_step = 5e-8
    grid_size = 1024
    field_size = 0.025

    with open(project_root / "configs/config.json") as config_file:
        config = json.load(config_file)

    train_loader = data.DataLoader(
        dataset=dataset.TrainData(
            sin_path=sin_path,
            theta=num_sensors,
            sample_N=sample_N,
            radius=radius,
            field_of_view=field_of_view,
            speed_of_sound=speed_of_sound,
            time_point=time_step,
            num_rays=num_rays,
            delay_sample=0,
            delta_t=1e-9,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    x_y = utils.setup_coordinate_grid(
        grid_size=grid_size,
        field_size=field_size,
    )
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    loss_function = torch.nn.MSELoss()
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=1,
        encoding_config=config["encoding"],
        network_config=config["network"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    epoch_bar = tqdm(
        range(num_epochs),
        desc="PA_INR",
        unit="epoch",
        colour="green",
    )
    for epoch in epoch_bar:
        model.train()
        loss_train = 0.0
        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=False,
            colour="blue",
        )
        for batch_index, (
            projection_l_sample,
            ray_plus,
            ray_minus,
            dr_sample_plus,
            dr_sample_minus,
            len_sample_plus,
            len_sample_minus,
        ) in enumerate(batch_bar):
            epoch_bar.set_description(
                f"PA_INR ==> Epoch:{epoch}/{num_epochs} "
                f"Sensor:{batch_index}/{len(train_loader)}"
            )

            current_batch_size = ray_plus.shape[0]
            ray_plus = ray_plus.to(device).float().view(-1, 2)
            ray_minus = ray_minus.to(device).float().view(-1, 2)
            projection_l_sample = projection_l_sample.to(device).float()
            dr_sample_plus = dr_sample_plus.to(device).float()
            dr_sample_minus = dr_sample_minus.to(device).float()
            len_sample_plus = len_sample_plus.to(device).float()
            len_sample_minus = len_sample_minus.to(device).float()

            intensity_minus = model(ray_minus).view(
                current_batch_size,
                sample_N,
                num_rays,
                1,
            )
            intensity_plus = model(ray_plus).view(
                current_batch_size,
                sample_N,
                num_rays,
                1,
            )
            projection_plus = 0.5 * (
                intensity_plus[:, :, :-1, :].sum(dim=2)
                + intensity_plus[:, :, 1:, :].sum(dim=2)
            ).squeeze(-1)
            projection_minus = 0.5 * (
                intensity_minus[:, :, :-1, :].sum(dim=2)
                + intensity_minus[:, :, 1:, :].sum(dim=2)
            ).squeeze(-1)

            predicted_projection = (
                projection_plus * dr_sample_plus / len_sample_plus
                - projection_minus * dr_sample_minus / len_sample_minus
            ) * 10

            loss = loss_function(projection_l_sample, predicted_projection)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            batch_bar.set_postfix(
                Lr=f"{scheduler.get_last_lr()[0]:.6f}",
                Loss=f"{loss.item():.4f}",
            )

        with torch.no_grad():
            reconstruction = (
                model(x_y.to(device).float().view(-1, 2))
                .view(grid_size, grid_size)
                .detach()
                .cpu()
                .numpy()
            )
        plt.imsave(
            result_dir / "latest_recon.png",
            reconstruction,
            cmap="gray",
        )

        scheduler.step()
        mean_loss = loss_train / len(train_loader)
        writer.add_scalar("MSE_train", mean_loss, epoch + 1)
        epoch_bar.set_postfix(last_epoch_loss=f"{mean_loss:.4f}")

    plt.imsave(
        result_dir / "final_recon.png",
        reconstruction,
        cmap="gray",
    )
    writer.close()
