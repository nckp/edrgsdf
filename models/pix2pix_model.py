import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A1', 'real_A2', 'real_A3', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
            self.model_names = ['G']
        self.netG = networks.define_G(3, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(6, self.opt.ndf, self.opt.netD,
                              self.opt.n_layers_D, self.opt.norm, self.opt.init_type, self.opt.init_gain,
                              self.gpu_ids)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.old_lr = opt.lr
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.n_epochs_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    # def set_input(self, input):
    #     AtoB = self.opt.direction == 'AtoB'
    #     self.real_A = input['A' if AtoB else 'B'].to(self.device)
    #     self.real_B = input['B' if AtoB else 'A'].to(self.device)
    #     self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A1 = input['A1' if AtoB else 'B1'].to(self.device).float()
        self.real_A2 = input['A2' if AtoB else 'B2'].to(self.device).float()
        self.real_A3 = input['A3' if AtoB else 'B3'].to(self.device).float()
        self.seed = input['seed'].to(self.device).float()
        self.seed = self.seed.view(self.seed.size(0), -1, 1, 1).expand(-1, -1, self.real_A1.size(2), self.real_A1.size(3))
        self.real_B = input['B' if AtoB else 'A'].to(self.device).float()
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        #self.real_A = torch.cat([self.real_A1, self.real_A2, self.real_A3, self.seed], dim=1)


    def forward(self):
        self.fake_B = self.netG(self.real_A1, self.real_A2, self.real_A3, self.seed)
        self.real_A = torch.cat((self.real_A1, self.fake_B), 1)

    def backward_D(self):
        fake_AB = torch.cat((self.real_A1, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        real_AB = torch.cat((self.real_A1, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A1, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
