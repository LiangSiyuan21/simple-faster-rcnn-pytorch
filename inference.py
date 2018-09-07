from __future__ import division
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import pdb
import torch
from torch.autograd import Variable
from torch.utils import data as data_
from PIL import Image
import argparse

WIDER_BBOX_LABEL_NAMES = (
    'Face')

parser = argparse.ArgumentParser(description='FaceShield Inference')
parser.add_argument('--ep', type=int, default=1,
                    help='Epsilon')

def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))

def inverse_normalize(img):
    return (img * 0.5 + 0.5).clip(min=0, max=1) * 255

def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    # normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                # std=[0.229, 0.224, 0.225])
    normalize = tvtsf.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
    img = normalize(t.from_numpy(img))
    return img.numpy()

def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
         (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray:
        A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    try:
        img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
    except:
        ipdb.set_trace()
    # both the longer and shorter should be less than
    # max_size and min_size
    normalize = pytorch_normalze
    return normalize(img)

def add_bbox(ax,bbox,label,score):
    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()
        label_names = list(WIDER_BBOX_LABEL_NAMES) + ['bg']
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax

class DCGAN(nn.Module):
	def __init__(self, num_channels=3, ngf=100, cg=0.05, learning_rate=1e-4, train_adv=False):
		"""
		Initialize a DCGAN. Perturbations from the GAN are added to the inputs to
		create adversarial attacks.

		- num_channels is the number of channels in the input
		- ngf is size of the conv layers
		- cg is the normalization constant for perturbation (higher means encourage smaller perturbation)
		- learning_rate is learning rate for generator optimizer
		- train_adv is whether the model being attacked should be trained adversarially
		"""
                super(DCGAN, self).__init__()
		self.generator = nn.Sequential(
			# input is (nc) x 32 x 32
			nn.Conv2d(num_channels, ngf, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2, inplace=True),
                        #nn.Dropout2d(),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2, inplace=True),
                        #nn.Dropout2d(),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2, inplace=True),
                        #nn.Dropout(),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2, inplace=True),
                        #nn.Dropout(),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 48 x 32 x 32
			nn.Conv2d(ngf, ngf, 1, 1, 0, bias=True),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. 3 x 32 x 32
			nn.Conv2d(ngf, num_channels, 1, 1, 0, bias=True),
			nn.Tanh()
		)

		self.cuda = torch.cuda.is_available()

		if self.cuda:
			self.generator.cuda()
                        self.generator = torch.nn.DataParallel(self.generator, device_ids=range(torch.cuda.device_count()))
			cudnn.benchmark = True

		# self.criterion = nn.NLLLoss()
		self.criterion = nn.CrossEntropyLoss(size_average=False)
		self.cg = cg
		self.optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
		self.train_adv = train_adv
                self.max_iter = 20
                self.c_misclassify = 1
                self.confidence = 0

	def forward(self, inputs, model, labels=None, bboxes=None, scale=None,\
                model_feats=None, model_optimizer=None, *args):
                """
                Given a set of inputs, return the perturbed inputs (as Variable objects),
                the predictions for the inputs from the model, and the percentage of inputs
                unsucessfully perturbed (i.e., model accuracy).

		If self.train_adversarial is True, train the model adversarially.

                The adversarial inputs is a python list of tensors.
                The predictions is a numpy array of classes, with length equal to the number of inputs.
                """
                num_unperturbed = 10
                iter_count = 0
                loss_perturb = 20
                loss_misclassify = 10
                while loss_misclassify > 0 and loss_perturb > 1:
                    perturbation = self.generator(inputs)
                    adv_inputs = inputs + perturbation
                    adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
                    scores,gt_labels  = model(adv_inputs,\
                            bboxes,labels,scale,attack=True)
                    probs = F.softmax(scores)
                    suppress_labels,probs,mask = model.faster_rcnn._suppress(None,probs,attack=True)
                    scores = scores[mask]
                    gt_labels = gt_labels[mask]
                    self.optimizer.zero_grad()
                    try:
                        one_hot_labels = torch.zeros(gt_labels.size() + (2,))
                        if self.cuda: one_hot_labels = one_hot_labels.cuda()
                        one_hot_labels.scatter_(1, gt_labels.unsqueeze(1).data, 1.)
                        labels_vars = Variable(one_hot_labels, requires_grad=False)
                        real = (labels_vars * scores).sum(1)
                        other = ((1. - labels_vars) * scores - labels_vars * 10000.).max(1)[0]

                        # the greater the likelihood of label, the greater the loss
                        loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
                        loss_misclassify = self.c_misclassify*torch.sum(loss1)
                        loss_match =  Variable(torch.zeros(1)).cuda()
                        loss_perturb = self.cg*L2_dist(inputs,adv_inputs)
                        loss_total = loss_misclassify + loss_perturb
                        loss_total.backward()
                        self.optimizer.step()
                    except:
                        loss_misclassify = Variable(torch.zeros(1)).cuda()
                        loss_match =  Variable(torch.zeros(1)).cuda()
                        loss_perturb = self.cg*L2_dist(inputs,adv_inputs)
                        loss_total = loss_misclassify + loss_perturb
                        loss_total.backward()

                    print('Loss NLL is %f, perturb %f, total loss %f' % \
                            (loss_misclassify.data,loss_perturb.data,loss_total.data))
                    # optimizer step for the generator

                    if loss_misclassify.data !=0:
                        predictions = torch.max(F.log_softmax(scores), 1)[1].cpu().numpy()
                        num_unperturbed = (predictions == gt_labels).sum()
                        print("Number of images unperturbed is %d out of %d" % \
                                (num_unperturbed,len(gt_labels)))
                    iter_count = iter_count + 1
                    losses = [Variable(loss_misclassify.data),Variable(torch.zeros(1)).cuda(),Variable(loss_perturb.data)]
                    losses = losses + [sum(losses)]
                    if iter_count > self.max_iter:
                        break
                return losses

	def perturb(self, inputs, epsilon=1, save_perturb=None):
		perturbation = self.generator(inputs)
		adv_inputs = inputs + epsilon*perturbation
		adv_inputs = torch.clamp(adv_inputs, -1.0, 1.0)
                if save_perturb is not None:
                    clamped = torch.clamp(perturbation,-1.0,1.0)
                    return adv_inputs,clamped
                else:
                    return adv_inputs

	def save(self, fn):
		torch.save(self.generator.state_dict(), fn)

	def save_cpu(self, fn):
		torch.save(self.generator.cpu().state_dict(), fn)

	def load(self, fn):
		self.generator.load_state_dict(torch.load(fn))

def model_fn(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = attacks.DCGAN(train_adv=False)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

if __name__ == '__main__':
    args = parser.parse_args()
    attacker = DCGAN(train_adv=False)
    attacker.load('min_max_attack.pth')
    img = read_image('stock_1.jpg')
    img = preprocess(img)
    img = torch.from_numpy(img)[None]
    im_path = 'stock_1.jpg'
    img = Variable(img.float().cuda())
    im_path_clone = b = '%s' % im_path
    ori_img_ = inverse_normalize(at.tonumpy(img[0]))
    adv_img = attacker.perturb(img,epsilon=args.ep)
    adv_img_ = inverse_normalize(at.tonumpy(adv_img[0]))
    ori_img_ = ori_img_.transpose((1,2,0))
    adv_img_ = adv_img_.transpose((1,2,0))

