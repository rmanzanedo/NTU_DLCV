import params
# from core import eval_src, eval_tgt, train_src, train_tgt
from models2 import Discriminator, LeNetClassifier, LeNetEncoder
# from utils import get_data_loader, init_model, init_random_seed
from sklearn.metrics import accuracy_score
import os
import data
import torch
import torch.optim as optim
from torch import nn



def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            result = my_net(input_data=imgs, mode='source', rec_scheme='share')
            pred = result[3]#.data.max(1, keepdim=True)[1]
            
            _, pred = torch.max(pred, dim = 1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()
            
            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)
    # print(gts)
    # print(preds)
    # exit()
   
    return accuracy_score(gts, preds)


def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)

def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net
def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)

def train_src(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(args.epch):
        # for step, (images, labels) in enumerate(data_loader):
        for idx, (imgs_src, imgs_tgt, cls) in enumerate(data_loader):

        ###################################
        # target data training            #
        ###################################

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(data_loader))
            # make images and labels variable
            images = make_variable(imgs_src)
            labels = make_variable(cls.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()
            print(train_info, end='\r')
            # print step info
            # if ((step + 1) % params.log_step_pre == 0):
            #     print("Epoch [{}/{}] Step [{}/{}]: loss={}"
            #           .format(epoch + 1,
            #                   params.num_epochs_pre,
            #                   step + 1,
            #                   len(data_loader),
            #                   loss.data[0]))

        # eval model on test set
        if epoch%args.val_epoch == 0:
            print(train_info)
            eval_src(encoder, classifier, data_loader)
        # if epoch%args.val_epoch == 0:
                ''' evaluate the model '''
                # acc = evaluate(encoder, classifier, data_loader)        
                # writer.add_scalar('val_acc', acc, iters)
                   
                # print('Epoch: [{}] ACC:{}'.format(epoch, acc))

        # save model parameters
        # if ((epoch + 1) % params.save_step_pre == 0):
        #     save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
        #     save_model(
        #         classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "log/ADDA-source-encoder-final.pt")
    save_model(classifier, "log/ADDA-source-classifier-final.pt")

    return encoder, classifier

def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for idx, (imgs_src, imgs_tgt, cls) in enumerate(data_loader):
        images = make_variable(imgs_src, volatile=True)
        labels = make_variable(cls)

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))




def train_tgt(src_encoder, tgt_encoder, critic,data_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    # len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        # zip source and target data pair
        # data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        # for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################
        for idx, (imgs_src, imgs_tgt, cls) in enumerate(data_loader):

        ###################################
        # target data training            #
        ###################################

            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(data_loader))

            # make images variable
            images_src = make_variable(imgs_src)
            images_tgt = make_variable(imgs_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            # if ((step + 1) % params.log_step == 0):
            #     print("Epoch [{}/{}] Step [{}/{}]:"
            #           "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
            #           .format(epoch + 1,
            #                   params.num_epochs,
            #                   step + 1,
            #                   len_data_loader,
            #                   loss_critic.data[0],
            #                   loss_tgt.data[0],
            #                   acc.data[0]))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):

            eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)

            
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
    return tgt_encoder



if __name__ == '__main__':
    # init random seed
    args = parser.arg_parse()
    init_random_seed(params.manual_seed)

    # load dataset
    # src_data_loader = get_data_loader(params.src_dataset)
    # src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    # tgt_data_loader = get_data_loader(params.tgt_dataset)
    # tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    src_data_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train', dataset= args.dataset_train, adaptation=args.adaptation),
                                           batch_size=args.train_batch, 
                                           num_workers=args.workers,
                                           shuffle=False)
    tgt_data_loader_eval  = torch.utils.data.DataLoader(data.DATA(args, mode='test', dataset=args.dataset_test, adaptation='0'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)

    # load models
    src_encoder = init_model(net=LeNetEncoder(),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(),
                                restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=LeNetEncoder(),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    # train source model
    print("=== Training classifier for source domain ===")
    # print(">>> Source Encoder <<<")
    # print(src_encoder)
    # print(">>> Source Classifier <<<")
    # print(src_classifier)

    # if not (src_encoder.restored and src_classifier.restored and
    #         params.src_model_trained):
    src_encoder, src_classifier = train_src(src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    # eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    # print(tgt_encoder)
    print(">>> Critic <<<")
    # print(critic)

    # init weights of target encoder with those of source encoder
    # if not tgt_encoder.restored:
    #     tgt_encoder.load_state_dict(src_encoder.state_dict())

    # if not (tgt_encoder.restored and critic.restored and
    #         params.tgt_model_trained):
    tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,src_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)