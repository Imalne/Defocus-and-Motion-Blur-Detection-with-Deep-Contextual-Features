from DataSetLoader.MDataSet import BlurDataSet
# from Net import Net, Net_Bn
from models import Net
from loss import HybridLoss
from torch.utils.data import DataLoader
from torch import optim
import torch
import os
from cfg import Configs
from tensorboardX import SummaryWriter


def valid(net, loader, loss_function, ech, summary):
    valid_loss = 0
    image = None
    gt1 = None
    gt2 = None
    gt3 = None
    gt4 = None
    pre_mask1 = None
    pre_mask2 = None
    pre_mask3 = None
    pre_mask4 = None
    for batch_id, data in enumerate(loader):
        input_image = torch.cat(data[0], 0).cuda(device=Configs["device_ids"][0])
        image = data[0][0][0]
        target = []
        for i in range(4):
            cur = []
            for t in data[1]:
                cur.append(t[i])
            target.append(torch.cat(cur, 0).cuda(device=Configs["device_ids"][0]))

        gt1 = data[1][0][0][0]
        gt2 = data[1][0][1][0]
        gt3 = data[1][0][2][0]
        gt4 = data[1][0][3][0]
        output = net(input_image)
        pre_mask1 = output[0].cpu()[0]
        pre_mask1 = torch.argmax(pre_mask1, dim=0).unsqueeze(0)
        pre_mask2 = output[1].cpu()[0]
        pre_mask2 = torch.argmax(pre_mask2, dim=0).unsqueeze(0)
        pre_mask3 = output[2].cpu()[0]
        pre_mask3 = torch.argmax(pre_mask3, dim=0).unsqueeze(0)
        pre_mask4 = output[3].cpu()[0]
        pre_mask4 = torch.argmax(pre_mask4, dim=0).unsqueeze(0)
        valid_loss, ce_loss, ssim_loss = loss_function(output, target)
        break
    summary.add_scalar("validate/total_loss", valid_loss.cpu().detach().numpy(), global_step=ech)
    summary.add_scalar("validate/ce_loss", ce_loss.cpu().detach().numpy(), global_step=ech)
    summary.add_scalar("validate/ssim_loss", ssim_loss.cpu().detach().numpy(), global_step=ech)
    summary.add_image("scalar/validate_sample_image", image, global_step=ech)
    summary.add_image("gt/validate_sample_gt1", gt1.unsqueeze(0), global_step=ech)
    summary.add_image("gt/validate_sample_gt2", gt2.unsqueeze(0), global_step=ech)
    summary.add_image("gt/validate_sample_gt3", gt3.unsqueeze(0), global_step=ech)
    summary.add_image("gt/validate_sample_gt4", gt4.unsqueeze(0), global_step=ech)
    summary.add_image("pre/validate_sample_pre1", pre_mask1, global_step=ech)
    summary.add_image("pre/validate_sample_pre2", pre_mask2, global_step=ech)
    summary.add_image("pre/validate_sample_pre3", pre_mask3, global_step=ech)
    summary.add_image("pre/validate_sample_pre4", pre_mask4, global_step=ech)
    return valid_loss


def train(net, train_loader, valid_loader, loss_function, opt, ech, summary):
    net.train()
    for batch_id, data in enumerate(train_loader):
        input_image = torch.cat(data[0], 0).cuda(device=Configs["device_ids"][0])
        target = []
        for i in range(4):
            cur = []
            for t in data[1]:
                cur.append(t[i])
            target.append(torch.cat(cur, 0).cuda(device=Configs["device_ids"][0]))

        opt.zero_grad()
        output = net(input_image)
        total_loss, ce_loss, ssim_loss = loss_function(output, target)
        total_loss.backward()
        opt.step()

        if batch_id % 10 == 0:
            valid_loss = valid(net, valid_loader,
                               loss_function,
                               ech * len(train_loader.dataset) + batch_id * train_loader.batch_size, summary)
            summary.add_scalar("train/total_loss", total_loss, global_step=ech)
            summary.add_scalar("train/ce_loss", ce_loss, global_step=ech)
            summary.add_scalar("train/ssim_loss", ssim_loss, global_step=ech)

            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  '\t train_loss: {:.12f} '
                  '\t ce_loss: {:.12f} '
                  '\t ssim_loss: {:.12f} '
                  '\t valid_loss: {:.12f}'.format(
                ech,
                batch_id * train_loader.batch_size,
                len(train_loader.dataset),
                100. * batch_id / len(train_loader),
                total_loss.data.cpu().numpy(),
                ce_loss.data.cpu().numpy(),
                ssim_loss.data.cpu().numpy(),
                valid_loss
            ))


def optimizer_by_layer(net, encoder_lr, decoder_lr_scale):
    params = [
        {"params": net.encoder.parameters(), "lr": encoder_lr},
        {"params": net.decoder.parameters(), "lr": encoder_lr * decoder_lr_scale},
        {"params": net.skip_1.parameters(), "lr": encoder_lr},
        {"params": net.skip_2.parameters(), "lr": encoder_lr},
        {"params": net.skip_3.parameters(), "lr": encoder_lr},
        {"params": net.skip_4.parameters(), "lr": encoder_lr}
    ]
    return optim.Adam(params=params, lr=encoder_lr)


if __name__ == '__main__':
    model = Net(Configs)

    model = torch.nn.DataParallel(model, device_ids=Configs["device_ids"])
    model = model.cuda(device=Configs["device_ids"][0])
    cur_epoch = model.module.load_model(Configs['vgg_19_pre_path'], Configs['model_save_path'])
    optimizer = optimizer_by_layer(model.module, Configs['encoder_learning_rate'], Configs['decoder_lr_scale'])

    train_data = BlurDataSet(Configs['train_image_dir'], Configs['train_mask_dir'], aug=Configs['augmentation'])
    train_loader = DataLoader(train_data, batch_size=Configs['train_batch_size'] * len(Configs['device_ids']),
                              shuffle=True, num_workers=len(Configs['device_ids']))
    test_data = BlurDataSet(Configs['test_image_dir'], Configs['test_mask_dir'], False)
    test_loader = DataLoader(test_data, batch_size=Configs['test_batch_size'] * len(Configs['device_ids']),
                             shuffle=True, num_workers=len(Configs['device_ids']))

    write = SummaryWriter()
    # write.add_graph(model,torch.rand(1,3,224,224).cuda())
    loss_func = HybridLoss(Configs["l_bce"], Configs["l_ssim"])
    for epoch in range(cur_epoch, Configs['epoch']):
        train(model, train_loader, test_loader, loss_func, optimizer, epoch, write)
        model.module.save_model(epoch, Configs['model_save_path'])
    write.close()
    exit(0)
