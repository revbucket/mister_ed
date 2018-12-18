
import torch.optim as optim
from torch.autograd import Variable
from cifar10 import cifar_loader as cl
from loss_functions import NFLoss
import adversarial_perturbations as ap
import adversarial_attacks as aa
import utils.checkpoints as checkpoints
import utils.pytorch_utils as utils
import time
import logging
from loss_functions import RegularizedLoss
from loss_functions import PartialXentropy


def train(batch_size=48):
    # set random seed for reproducibility
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)  # ignore this if using CPU

    # Match the normalizer using in the official implementation
    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5],
                                               std=[1.0, 1.0, 1.0])

    # use resnet32
    classifier_net = cl.load_pretrained_cifar_resnet(flavor=32,
                                                     return_normalizer=False,
                                                     manual_gpu=None)
    # load cifar data
    cifar_train = cl.load_cifar_data('train', batch_size=batch_size)
    cifar_test = cl.load_cifar_data('train', batch_size=256)

    nfp = NeuralFP(classifier_net=classifier_net, num_dx=5, num_class=10, dataset_name="cifar",
                   log_dir="~/Documents/deep_learning/AE/submit/mister_ed/pretrained_model")

    num_epochs = 30
    verbosity_epoch = 5

    train_loss = nn.CrossEntropyLoss()

    logger = nfp.train(cifar_train, cifar_test, normalizer, num_epochs, train_loss,
                       verbosity_epoch)

    return logger


def test():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)  # ignore this if using CPU

    # Match the normalizer using in the official implementation
    # TODO: Check Normalizer's effect
    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5],
                                               std=[1.0, 1.0, 1.0])

    # get the model
    classifier_net = CW2_Net()
    print("Eval using model", classifier_net)

    # load the weight
    PATH = "/home/tianweiy/Documents/deep_learning/AE/NeuralFP/NFP_model_weights/cifar/eps_0.006/numdx_30/ckpt" \
           "/state_dict-ep_80.pth"
    classifier_net.load_state_dict(torch.load(PATH))
    classifier_net.cuda()
    classifier_net.eval()
    print("Loading checkpoint")

    # Original Repo uses pin memory here
    cifar_test = cl.load_cifar_data('val', shuffle=False, batch_size=64)
    normalizer = utils.DifferentiableNormalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])

    # restore fingerprints
    fingerprint_dir = "/home/tianweiy/Documents/deep_learning/AE/NeuralFP/NFP_model_weights/cifar/eps_0.006/numdx_30/"

    # fixed_dxs = pandas.read_pickle(os.path.join(fingerprint_dir, "fp_inputs_dx.pkl"))
    # fixed_dys = pandas.read_pickle(os.path.join(fingerprint_dir, "fp_outputs.pkl"))

    fixed_dxs = np.load(os.path.join(fingerprint_dir, "fp_inputs_dx.pkl"), encoding='bytes')
    fixed_dys = np.load(os.path.join(fingerprint_dir, "fp_outputs.pkl"), encoding='bytes')

    # print(fixed_dxs)
    # print(fixed_dys)

    fixed_dxs = utils.np2var(np.concatenate(fixed_dxs, axis=0), cuda=True)
    fixed_dys = utils.np2var(fixed_dys, cuda=True)

    # print(fixed_dxs.shape)
    # print(fixed_dys.shape)

    reject_thresholds = \
        [0. + 0.001 * i for i in range(0, 2000)]

    print("Dataset CIFAR")

    loss = NFLoss(classifier_net, num_dx=30, num_class=10, fp_dx=fixed_dxs, fp_target=fixed_dys, normalizer=normalizer)

    logger = logging.getLogger('sanity')
    hdlr = logging.FileHandler('/home/tianweiy/Documents/deep_learning/AE/NeuralFP/log/pgd_2000_16_5_testV3.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.WARNING)

    # sanity check all clean examples are valid
    print("USE PGD 2000 STEP")
    for weight in [1., 10., 100., 1000., 10000.]:
        logger.exception("Use Weight " + str(weight))

        dis_adv = []
        dis_real = []

        for idx, test_data in enumerate(cifar_test, 0):
            inputs, labels = test_data

            inputs = inputs[0].unsqueeze(0)
            labels = labels[0].unsqueeze(0)

            inputs = inputs.cuda()  # comment this if using CPU
            labels = labels.cuda()

            # build adversarial example
            delta_threat = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 'inf',
                                                             'lp_bound': 16.0 / 255})

            vanilla_loss = PartialXentropy(classifier_net, normalizer)
            losses = {'vanilla': vanilla_loss, 'fingerprint': loss}
            scalars = {'vanilla': 1., 'fingerprint': -1 * weight}

            attack_loss = RegularizedLoss(losses=losses, scalars=scalars)

            pgd_attack_object = aa.PGD(classifier_net, normalizer, delta_threat, attack_loss)
            perturbation_out = pgd_attack_object.attack(inputs, labels, num_iterations=2000, verbose=False)
            adv_examples = perturbation_out.adversarial_tensors()

            assert adv_examples.size(0) is 1

            # compute adversarial loss
            l_adv = loss.forward(adv_examples, labels)
            loss.zero_grad()

            # compute real image loss
            l_real = loss.forward(inputs, labels)
            loss.zero_grad()

            dis_adv.append(l_adv)
            dis_real.append(l_real)
            # if idx % 1000 == 0:
            #    print("FINISH", idx, "EXAMPLES")
            #    print("Adversarial Percent is ", adversarial / total * 100, "%")
            #    print("False Positive is ", fpositive / total * 100, "%")

        total = len(dis_adv)

        for tau in reject_thresholds:
            true_positive = 0
            false_positive = 0

            for adv in dis_adv:
                if adv > tau:
                    true_positive += 1

            for real in dis_real:
                if real > tau:
                    false_positive += 1

            logger.exception("The Threshold is "+str(tau))
            logger.exception("True Positive is " + str(true_positive / total * 100) + '%')
            logger.exception("False Positive is " + str(false_positive / total * 100) + '%')
            # print("The Accuracy on Clean Example is ", correct / total * 100, "%")






if __name__ == '__main__':
    test()