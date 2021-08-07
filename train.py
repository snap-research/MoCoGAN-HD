"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import time
import os

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from util.visualizer import Visualizer
from models.models import create_model


def main():
    args = TrainOptions().parse()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.world_batch_size = args.batchSize
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu_ids, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    if args.cross_domain:
        import train_func_cross_domain as train_func
    else:
        import train_func_in_domain as train_func
    args.gpu = gpu
    torch.backends.cudnn.benchmark = True

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        __builtins__['print'] = print_pass

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            args.batchSize = int(args.batchSize / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)

    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    modelG, modelD_img, modelD_3d = create_model(args)

    data_loader = CreateDataLoader(args)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# training videos = %d' % dataset_size)

    visualizer = None
    if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        visualizer = Visualizer(args)

    x = torch.FloatTensor(args.batchSize, args.n_frames_G, args.nc,
                          args.video_frame_size, args.video_frame_size)
    z = torch.FloatTensor(args.batchSize, args.latent_dimension)
    z_fix = torch.FloatTensor(args.batchSize, args.latent_dimension)

    if args.gpu is not None:
        x = x.cuda(args.gpu)
        z = z.cuda(args.gpu)
        z_fix = z_fix.cuda(args.gpu)
    else:
        x = x.cuda()
        z = z.cuda()
        z_fix = z_fix.cuda()
    z_fix.data.normal_()
    if args.rank % ngpus_per_node == 0:
        writer = SummaryWriter(
            log_dir=os.path.join(args.checkpoints_dir, 'runs'))

    total_steps = 0
    train_func.toggle_grad(modelG, False)

    for epoch in range(args.load_pretrain_epoch + 1, args.total_epoch):
        if args.distributed:
            data_loader.train_sampler.set_epoch(epoch)

        for step, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += 1
            loss_all, loss_names = train_func.GD_step(args, modelG, modelD_img,
                                                      modelD_3d, data, x, z)
            loss_dict = dict(zip(loss_names, loss_all))

            if total_steps % args.print_freq == 0 and (
                    not args.multiprocessing_distributed or
                (args.multiprocessing_distributed
                 and args.rank % ngpus_per_node == 0)):
                t = time.time() - iter_start_time
                errors = {k: v for k, v in loss_dict.items()}
                visualizer.print_current_errors(epoch, step, errors, t)

            if total_steps % args.save_latest_freq == 0 and (
                    not args.multiprocessing_distributed or
                (args.multiprocessing_distributed
                 and args.rank % ngpus_per_node == 0)):
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                save_models(modelG, modelD_img, modelD_3d,
                            args.checkpoints_dir, 'latest')

        if epoch % args.save_epoch_freq == 0 and (
                not args.multiprocessing_distributed or
            (args.multiprocessing_distributed
             and args.rank % ngpus_per_node == 0)):
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            save_models(modelG, modelD_img, modelD_3d, args.checkpoints_dir,
                        'epoch_' + str(epoch))
            save_videos(writer, z_fix, modelG, 'epoch_' + str(epoch), args)
    if args.rank % ngpus_per_node == 0:
        writer.close()


def save_models(modelG, modelD_img, modelD_3d, ckpt_dir, string):
    torch.save(modelG.module.modelR.state_dict(),
               '%s/modelR_%s.pth' % (ckpt_dir, string))
    torch.save(modelD_img.state_dict(),
               '%s/modelD_img_%s.pth' % (ckpt_dir, string))
    torch.save(modelD_3d.state_dict(),
               '%s/modelD_3d_%s.pth' % (ckpt_dir, string))


def save_videos(writer, z, modelG, string, args):
    with torch.no_grad():
        modelG.eval()
        x_fake, _, _ = modelG([z], args.n_frames_G, use_noise=True)
        x_fake = x_fake.view(args.batchSize, args.n_frames_G, args.nc,
                             args.style_gan_size, args.style_gan_size)
        x_fake = x_fake.clamp(-1, 1)
        writer.add_video(string, (x_fake.data + 1.) / 2.)
        modelG.train()


if __name__ == "__main__":
    main()
