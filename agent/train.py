import torch
import numpy as np
import torchvision

from .models import Detector, save_model, FocalLoss, load_model
from .utils import load_detection_data, accuracy
import torch.utils.tensorboard as tb


def train(args):
    # from os import path
    # model = Detector()
    # train_logger, valid_logger = None, None
    # if args.log_dir is not None:
    #     train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
    #     valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() and not (args.cpu or args.debug) else 'cpu')
    print(device)
    model = None
    best_loss = 100

    # Hyperparameters
    lrs = [1e-3]
    optimizers = ["adam"]
    n_epochs = 400
    batch_size = 32
    num_workers = 0 if args.debug else 4
    # properties = [(True, True, True)]
    properties = [(False, True, True)]
    # dimensions = [[64, 128, 256]]
    dimensions = [[32, 64, 128]]
    gammas = [2]
    # loss_sizes_weight = [0.01]
    loss_sizes_weight = [0.1]

    # loader_valid = load_detection_data('dense_data/valid', num_workers=num_workers, batch_size=batch_size, drop_last=False)
    transforms = [
        torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)), torchvision.transforms.ToTensor()]),
        # torchvision.transforms.Compose([torchvision.transforms.Resize((240, 240)), torchvision.transforms.ToTensor()])
        # torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
        #                                 torchvision.transforms.Resize((128, 128)), torchvision.transforms.ToTensor()]),
    ]
    for t in range(len(transforms)):
        # Get data
        loader_train = load_detection_data('data', num_workers=num_workers, batch_size=batch_size,
                                           transform=transforms[t])

        for size_weight in loss_sizes_weight:
            for gamma in gammas:
                # Initialize loss
                if gamma == 0:
                    loss_centers = torch.nn.BCEWithLogitsLoss().to(device)
                else:
                    loss_centers = FocalLoss(gamma=gamma, reduce=True).to(device)
                loss_sizes = torch.nn.MSELoss(reduction='none').to(device)

                for scheduler_type in ["minLoss"]:
                    for dim in dimensions:
                        for prop in properties:
                            for optim_name in optimizers:
                                for lr in lrs:
                                    # Tensorboard
                                    global_step = 0
                                    name_model = f"{t + 3}/{optim_name}/{lr}/{dim}/" \
                                                 f"residual&skip&inputNorm={prop[0]}&{prop[1]}&{prop[2]}/{scheduler_type}/" \
                                                 f"gamma={gamma}/size_weight={size_weight}/"
                                    train_logger = tb.SummaryWriter(f"{args.log_dir}/train/{name_model}")
                                    # valid_logger = tb.SummaryWriter(f"{args.log_dir}/valid/{name_model}")

                                    del model
                                    model = Detector(dim_layers=dim, residual=prop[0], skip_connections=prop[1],
                                                     input_normalization=prop[2])
                                    if args.continue_training:
                                        model = load_model()
                                    model = model.to(device)

                                    if optim_name == "sgd":
                                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                                                    weight_decay=1e-4)
                                    else:
                                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                                    if scheduler_type == "minLoss":
                                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                                               patience=5)
                                    else:
                                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                                                               patience=3)
                                    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

                                    print(f"{args.log_dir}/{name_model}")
                                    for epoch in range(n_epochs):
                                        print(epoch)
                                        train_loss = []
                                        train_size_loss = []
                                        train_classes_loss = []
                                        acc_sizes = []
                                        acc_pred = []

                                        # Start Training
                                        model.train()
                                        for img, label, sizes in loader_train:
                                            img = img.to(device)
                                            label = label.to(device)
                                            sizes = sizes.to(device)

                                            # Compute loss and update parameters
                                            pred, pred_sizes = model(img)
                                            loss_val_classes = loss_centers(pred, label)
                                            loss_val_sizes = loss_sizes(pred_sizes, sizes)

                                            # Combine losses
                                            loss_val_sizes = (loss_val_sizes * label[:, None]).mean()
                                            loss_val = loss_val_classes + size_weight * loss_val_sizes

                                            optimizer.zero_grad()
                                            loss_val.backward()
                                            optimizer.step()

                                            train_size_loss.append(loss_val_sizes.cpu().detach().numpy())
                                            train_classes_loss.append(loss_val_classes.cpu().detach().numpy())
                                            train_loss.append(loss_val.cpu().detach().numpy())
                                            acc_pred.append(accuracy(pred, label))
                                            acc_sizes.append(accuracy(pred_sizes, sizes))

                                        # scheduler.step()
                                        if scheduler_type == "minLoss":
                                            scheduler.step(np.mean(train_loss))
                                        # elif scheduler_type == "maxAcc":
                                        #     scheduler.step(valid_acc.global_accuracy)
                                        # else:
                                        #     scheduler.step(valid_acc.iou)

                                        global_step += 1
                                        if train_logger is not None:
                                            log(train_logger, img, label, pred, global_step)
                                            cur_acc_pred = np.mean(acc_pred)
                                            train_logger.add_scalar('acc_pred', cur_acc_pred,
                                                                    global_step=global_step)
                                            train_logger.add_scalar('acc_sizes', np.mean(acc_sizes),
                                                                    global_step=global_step)
                                            cur_train_loss = np.mean(train_loss)
                                            train_logger.add_scalar('loss', cur_train_loss,
                                                                    global_step=global_step)
                                            train_logger.add_scalar('loss_size', np.mean(train_size_loss),
                                                                    global_step=global_step)
                                            train_logger.add_scalar('loss_classes', np.mean(train_classes_loss),
                                                                    global_step=global_step)
                                            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                                                    global_step=global_step)

                                            # Save best model
                                            if cur_train_loss < best_loss:
                                                save_model(model, 'det_best.th')
                                                best_loss = cur_train_loss

                                        # Compute validation accuracy
                                        # if (epoch + 1) % 10 == 0:
                                        #     model.eval()
                                        #     ap = DetectionAP(model, device)
                                        #
                                        #     if scheduler_type != "minLoss":
                                        #         scheduler.step(ap.test_box_ap0)
                                        #
                                        #     if train_logger is not None:
                                        #         valid_logger.add_scalar('dist0', ap.test_dist_ap0,
                                        #                                 global_step=global_step)
                                        #         valid_logger.add_scalar('dist1', ap.test_dist_ap1,
                                        #                                 global_step=global_step)
                                        #         valid_logger.add_scalar('dist2', ap.test_dist_ap2,
                                        #                                 global_step=global_step)
                                        #         valid_logger.add_scalar('box0', ap.test_box_ap0,
                                        #                                 global_step=global_step)
                                        #         valid_logger.add_scalar('box1', ap.test_box_ap1,
                                        #                                 global_step=global_step)
                                        #         valid_logger.add_scalar('box2', ap.test_box_ap2,
                                        #                                 global_step=global_step)
                                        #         valid_logger.add_scalar('iou0', ap.test_iou_ap0,
                                        #                                 global_step=global_step)
                                        #         valid_logger.add_scalar('iou1', ap.test_iou_ap1,
                                        #                                 global_step=global_step)
                                        #         valid_logger.add_scalar('iou2', ap.test_iou_ap2,
                                        #                                 global_step=global_step)

    save_model(model, 'det_final.th')


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16, None], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16, None]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default="./logs")
    # Put custom arguments here
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
