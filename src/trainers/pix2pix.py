import os
import time
import torch
from tqdm import tqdm

def train_p2p(G, D, train_loader, val_loader, criterion, criterionL1, optimizer_G, optimizer_D, epochs, lambda_L1=100, scheduler=None, start_epoch=None, metrics=None, device='cuda'):

    history = {
        'train_losses_G': [],
        'train_losses_D': [],
        'val_losses_G': [],
        'val_losses_D': [],
        'train_time': [],
        'val_time': [],
        }

    if metrics is not None:
        for metric_name in metrics.keys():
            history[f'train_{metric_name}'] = []
            history[f'val_{metric_name}'] = []

    if scheduler is not None:
        history['lr'] = []

    if start_epoch is None:
        start_epoch = 1

    if start_epoch > epochs:
        raise ValueError("start_epoch must be less than or equal to epochs")

    for epoch in range(start_epoch, epochs + 1):
        train_p_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        epoch_train_start_time = time.time()

        epoch_train_loss_G = 0
        epoch_train_loss_D = 0
        epoch_val_loss_G = 0
        epoch_val_loss_D = 0

        if metrics is not None:
            epoch_train_metrics = {metric_name: 0 for metric_name in metrics.keys()}
            epoch_val_metrics = {metric_name: 0 for metric_name in metrics.keys()}

        G.train()
        D.train()
        for _, (real_A, real_B) in train_p_bar:
            real_A, real_B = real_A.to(device), real_B.to(device)

            # Forward pass
            fake_B = G(real_A)

            # set requires grad
            for param in D.parameters():
                param.requires_grad = True
            optimizer_D.zero_grad()

            # backward D
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = D(fake_AB.detach())
            loss_D_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = D(real_AB)
            loss_D_real = criterion(pred_real, torch.ones_like(pred_real))
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # set requires grad
            for param in D.parameters():
                param.requires_grad = False
            optimizer_G.zero_grad()

            # backward G
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = D(fake_AB)
            loss_G_GAN = criterion(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step() 

            batch_size = real_A.size(0)
            epoch_train_loss_D += loss_D.item() * batch_size
            epoch_train_loss_G += loss_G.item() * batch_size

            if metrics is not None:
                real_B_detached = real_B.detach().cpu()
                fake_B_detached = fake_B.detach().cpu()
                for metric_name, metric_fn in metrics.items():
                    metric_value = metric_fn(real_B_detached, fake_B_detached)
                    epoch_train_metrics[metric_name] += metric_value * batch_size

            train_p_bar.set_description(f"Epoch: {epoch}/{epochs}, Loss G: {loss_G.item():.4f}, Loss D: {loss_D.item():.4f}")

        epoch_train_time = time.time() - epoch_train_start_time
        history['train_time'].append(epoch_train_time)

        epoch_train_loss_G /= len(train_loader.dataset)
        epoch_train_loss_D /= len(train_loader.dataset)

        history['train_losses_G'].append(epoch_train_loss_G)
        history['train_losses_D'].append(epoch_train_loss_D)

        if metrics is not None:
            for metric_name in metrics.keys():
                epoch_train_metrics[metric_name] /= len(train_loader.dataset)
                history[f'train_{metric_name}'].append(epoch_train_metrics[metric_name])

            train_metrics_formatted = ', '.join([f"{metric_name}: {epoch_train_metrics[metric_name]:.4f}" for metric_name in metrics.keys()])
            print(f"Train Metrics: Loss G: {epoch_train_loss_G:.4f}, Loss D: {epoch_train_loss_D:.4f}, {train_metrics_formatted}")
        else:
            print(f"Train Loss G: {epoch_train_loss_G:.4f}, Loss D: {epoch_train_loss_D:.4f}")
        

        
        G.eval()
        D.eval()
        with torch.inference_mode():
            val_p_bar = tqdm(enumerate(val_loader), total=len(val_loader))

            epoch_val_start_time = time.time()
            for _, (real_A, real_B) in val_p_bar:
                real_A, real_B = real_A.to(device), real_B.to(device)

                # Forward pass
                fake_B = G(real_A)

                # Discriminator loss for validation
                fake_AB = torch.cat((real_A, fake_B), 1)
                pred_fake = D(fake_AB)
                loss_D_fake = criterion(pred_fake, torch.zeros_like(pred_fake))
                real_AB = torch.cat((real_A, real_B), 1)
                pred_real = D(real_AB)
                loss_D_real = criterion(pred_real, torch.ones_like(pred_real))
                loss_D_val = (loss_D_fake + loss_D_real) * 0.5

                # Generator loss for validation
                loss_G_GAN = criterion(pred_fake, torch.ones_like(pred_fake))
                loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
                loss_G_val = loss_G_GAN + loss_G_L1 

                batch_size = real_A.size(0)

                epoch_val_loss_G += loss_G_val.item() * batch_size
                epoch_val_loss_D += loss_D_val.item() * batch_size

                if metrics is not None:
                    real_B_detached = real_B.detach().cpu()
                    fake_B_detached = fake_B.detach().cpu()
                    for metric_name, metric_fn in metrics.items():
                        metric_value = metric_fn(real_B_detached, fake_B_detached)
                        epoch_val_metrics[metric_name] += metric_value * batch_size

                train_p_bar.set_description(f"Epoch: {epoch}/{epochs}, Loss G: {loss_G.item():.4f}, Loss D: {loss_D.item():.4f}")
            
            epoch_val_time = time.time() - epoch_val_start_time
            history['val_time'].append(epoch_val_time)


            epoch_val_loss_G /= len(val_loader.dataset)
            epoch_val_loss_D /= len(val_loader.dataset)

            history['val_losses_G'].append(epoch_val_loss_G)
            history['val_losses_D'].append(epoch_val_loss_D)

            if metrics is not None:
                for metric_name in metrics.keys():
                    epoch_val_metrics[metric_name] /= len(val_loader.dataset)
                    history[f'val_{metric_name}'].append(epoch_val_metrics[metric_name])

                val_metrics_formatted = ', '.join([f"{metric_name}: {epoch_val_metrics[metric_name]:.4f}" for metric_name in metrics.keys()])
                print(f"Val Metrics: Loss G: {epoch_val_loss_G:.4f}, Loss D: {epoch_val_loss_D:.4f}, {val_metrics_formatted}")
            else:
                print(f"Val Loss G: {epoch_val_loss_G:.4f}, Loss D: {epoch_val_loss_D:.4f}")

        if scheduler is not None:
            scheduler.step()
            history['lr'].append(scheduler.get_last_lr()) 
            # print(f"Learning rate: {scheduler.get_last_lr()}")

        # if epoch % CHECKPOINT_STEP == 0:
        #     save_checkpoint(model, optimizer, epoch, history, os.path.join(MODELS_PATH, f'{MODEL_NAME}_model_{epoch}E.pth'))
        # else:
        #     print('\n')
    return history