import torch
from .base_model import BaseModel
from models.networks import networks,losses
from models.networks import resnet

class GANModel(BaseModel):
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
            parser.add_argument('--z_dist', type=str, default="gaussian", help='distrubtion of z. uniform | gaussian | zeros')
            parser.add_argument('--z_dim', type=int, default=256, help='dimension of z')
            parser.add_argument('--lambda_reg', type=float, default=10, help='lambda for the regularizer')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'D_reg']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = resnet.Generator(opt.z_dim, opt.crop_size)
        networks.init_net(self.netG, gpu_ids=self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = resnet.Discriminator(opt.z_dim, opt.crop_size)
            networks.init_net(self.netD, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=opt.lr, alpha=0.99, eps=1e-8)
            self.optimizer_D = torch.optim.RMSprop(self.netD.parameters(), lr=opt.lr, alpha=0.99, eps=1e-8)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # define the prior distribution
        if opt.z_dist == 'uniform':
            low = -torch.ones(opt.z_dim, device=self.device)
            high = torch.ones(opt.z_dim, device=self.device)
            self.z_dist = torch.distributions.Uniform(low, high)
        elif opt.z_dist == 'gaussian':
            mu = torch.zeros(opt.z_dim, device=self.device)
            scale = torch.ones(opt.z_dim, device=self.device)
            self.z_dist = torch.distributions.Normal(mu, scale)
        else:
            raise NotImplementedError


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']

        print(self.netG.module)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.z = self.z_dist.sample((self.opt.batch_size,))
        self.fake = self.netG(self.z)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Real
        self.real.requires_grad_()
        pred_real = self.netD(self.real)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_real.backward(retain_graph=True)
        # Gradient regularizer
        self.loss_D_reg = losses.compute_grad2(pred_real,self.real) * self.opt.lambda_reg
        self.loss_D_reg.backward()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_fake.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD( self.fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN
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
        self.optimizer_G.step()             # udpate G's weights
