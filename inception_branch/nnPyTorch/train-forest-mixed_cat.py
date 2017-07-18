
from dataset.kgforest import *
from dataset.tool import *

from net.rates import *
from net.util import *

from net.model.resnet_inception import ResNet_Inception as Net

import torch.nn.functional as F
import os.path

SIZE = 256
SRC = 'tif'
CH = 'rgb'
SEED = 123

use_gpu = True

# SRC = 'jpg' #channel
#CH = 'irrg'
# CH = 'irrgb'


def loss_func(logits, labels):
    loss = nn.MultiLabelSoftMarginLoss()(logits, Variable(labels))
    return loss


def multi_f_measure(probs, labels, threshold=0.235, beta=2):

    SMALL = 1e-6  # 1e-12
    batch_size = probs.size()[0]

    l = labels
    p = (probs > threshold).float()

    num_pos = torch.sum(p,  1)
    num_pos_hat = torch.sum(l,  1)
    tp = torch.sum(l * p, 1)
    precise = tp / (num_pos + SMALL)
    recall = tp / (num_pos_hat + SMALL)

    fs = (1 + beta * beta) * precise * recall / \
        (beta * beta * precise + recall + SMALL)
    f = fs.sum() / batch_size
    return f


def f2_score(y_pred, y_true, thres=0.235):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    y_pred = y_pred > thres
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def augment(x, u=0.75):
    if random.random() < u:
        if random.random() > 0.5:
            x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
        else:
            x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)
        x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10,
                                   rotate_limit=45, u=1)

        x = randomFlip(x, u=0.5)
        x = randomTranspose(x, u=0.5)
        # x = randomContrast(x, limit=0.2, u=0.5)
        # x = randomSaturation(x, limit=0.2, u=0.5),
        # x = randomFilter(x, limit=0.5, u=0.2)
    return x


def do_thresholds(probs, labels):
    print('\n--- [START %s] %s\n\n' %
          (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    print('\n')

    nClass = len(CLASS_NAMES)
    tryVals = np.arange(0, 1, 0.005)
    scores = np.zeros(len(tryVals))

    # single threshold
    for i, t in enumerate(tryVals):
        scores[i] = fbeta_score(labels, probs > t, beta=2, average='samples')

    best_single_thres = tryVals[scores.argmax()]
    best_single_thres_score = scores.max()

    print('*best_threshold (fixed)*\n')
    print("%0.4f" % best_single_thres)
    print('\n')
    print('*best_score*\n')
    print('%0.4f\n' % best_single_thres_score)

    # per class threshold
    best_thresholds = np.ones(nClass) * best_single_thres
    best_multi_thres_score = 0
    noChange = 0
    for iter in range(nClass * 10):
        print("thres scan iter %i" % iter)
        trial_thresholds = best_thresholds.copy()
        targetClass = iter % nClass
        for i, t in enumerate(tryVals):
            trial_thresholds[targetClass] = t
            scores[i] = fbeta_score(
                labels, probs > trial_thresholds, beta=2, average='samples')

        best_threshold = tryVals[scores.argmax()]
        best_multi_thres_score = scores.max()
        if best_threshold == best_thresholds[targetClass]:
            noChange += 1
        else:
            noChange = 0
            best_thresholds[targetClass] = best_threshold

        if noChange == nClass:
            break

    print('*best_threshold (per class)*\n')
    print(np.array2string(best_thresholds, formatter={
          'float_kind': lambda x: ' %.3f' % x}, separator=','))
    print('\n')
    print('*best_score*\n')
    print('%0.4f\n' % best_multi_thres_score)

    return best_single_thres, best_thresholds


def do_predict(net, dataset, batch_size=20, silent=True):
    if use_gpu:
        net.cuda().eval()
    else:
        net.eval()

    num_classes = len(CLASS_NAMES)
    logits = np.zeros((len(dataset), num_classes), np.float32)
    probs = np.zeros((len(dataset), num_classes), np.float32)

    tot_samples = 0

    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),  # None,
        batch_size=batch_size,
        # drop_last   = False,
        num_workers=0,
        pin_memory=True)

    for it, batch in enumerate(loader, 0):
        if not silent:
            print("predict batch %i / %i" % (it, len(loader)))
        # images = batch['tif'][:,1:,:,:] #IR R G B to R G B
        imagesRGB = batch['jpg']
        imagesNIR = batch['tif'][:, 0:1, :, :]
        if use_gpu:
            imagesRGB = imagesRGB.cuda()
            imagesNIR = imagesNIR.cuda()

        batch_size = len(imagesRGB)
        tot_samples += batch_size
        start = tot_samples - batch_size
        end = tot_samples

        # forward
        #ls, ps = net(Variable(images.cuda(),volatile=True))

        output_cat, probs_cat = net(
            Variable(imagesNIR, volatile=True),
            Variable(imagesRGB, volatile=True)
        )

        # print(probs.data.cpu().numpy().reshape(-1,num_classes))
        logits[start:end] = output_cat.data.cpu(
        ).numpy().reshape(-1, num_classes)
        probs[start:end] = probs_cat.data.cpu(
        ).numpy().reshape(-1, num_classes)

    assert(len(dataset) == tot_samples)

    return logits, probs


def do_submit(prob, thres, imgKeys, outfile="submit.csv"):
    tagsVec = probs > thres
    tagsCol = []
    for arow in tagsVec:
        tags = [CLASS_NAMES[i] for i in np.where(arow)[0]]
        tags = " ".join(tags)
        tagsCol.append(tags)
    output = pd.DataFrame()
    output['image_name'] = imgKeys
    output['tags'] = tagsCol
    output.to_csv(outfile, index=False)


def get_model(init_file_list=None):
    print('** net setting **\n')
    num_classes = len(CLASS_NAMES)
    net = Net(in_shape_inc=(1, SIZE, SIZE), in_shape_res=(
        3, SIZE, SIZE), num_classes=num_classes)
    print('%s\n\n Model Type:' % (type(net)))
    print('===== Combined models ======')
    print('\n')

    ''' Freezing some parts of the model'''
    for param in net.parameters():
        param.requires_grad = False

    trainable_layers = [net.fc]
    trainable_paras = []
    for aLayer in trainable_layers:
        for param in aLayer.parameters():
            param.requires_grad = True
            trainable_paras.append(param)

    optimizer = optim.Adam(net.fc.parameters())
    # optimizer = optim.SGD(train_param, lr=0.1, momentum=0.9,
    # weight_decay=0.0005)  # 0.0005

    # resume from previous ----------------------------------
    start_epoch = 0
    if init_file_list is not None:
        if len(init_file_list) == 2:
            init_content_inc = torch.load(init_file_list[0]).state_dict()
            init_content_res = torch.load(init_file_list[1]).state_dict()
            # load inception
            skip_list = ['fc.weight', 'fc.bias']
            load_model_weight(
                net.model_inc, init_content_inc, skip_list=skip_list)
            # load resnet
            skip_list = ['fc.weight', 'fc.bias']
            load_model_weight(
                net.model_res, init_content_res, skip_list=skip_list)
            # constrct fc from inception and resnet fc
            ratio_inc = 2048 / 2560
            ratio_res = 512 / 2560
            net.state_dict()['fc.weight'][:, 0:2048] = init_content_inc[
                'fc.weight'] * ratio_inc
            net.state_dict()['fc.weight'][:, 2048:] = init_content_res[
                'fc.weight'] * ratio_res
            net.state_dict()['fc.bias'] = (init_content_inc[
                'fc.bias'] * ratio_inc) + (init_content_res['fc.bias'] * ratio_res)

        if len(init_file_list) == 1:
            init_content = torch.load(init_file_list[0]).state_dict()
            load_model_weight(net, init_content)

    return net, optimizer, start_epoch


def do_training(out_dir='../../output/inception_and_resnet'):

    path_inc = '../../output/inception_tif_NIR_out/snap/best_acc_inception.torch'
    path_res = '../../output/best_acc_resnet34.torch'
    init_file_list = [path_inc, path_res]

    if os.path.exists('../../output/inception_and_resnet/snap/mixed_cat_best.torch'):
        init_file_list = [
            '../../output/inception_and_resnet/snap/mixed_cat_best.torch']
    # ------------------------------------
    if not os.path.exists(out_dir + '/snap'):
        os.makedirs(out_dir + '/snap')

    print('\n--- [START %s] %s\n\n' %
          (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    print('** some experiment setting **\n')
    print('\tSEED    = %u\n' % SEED)
    print('\tfile    = %s\n' % __file__)
    print('\tout_dir = %s\n' % out_dir)
    print('\n')

    # dataset ----------------------------------------
    print('** dataset setting **\n')
    num_classes = len(CLASS_NAMES)
    batch_size = 20  # 48  #96 #96  #80 #96 #96   #96 #32  #96 #128 #

    train_dataset = KgForestDataset('train_35479.txt',
                                    # train_dataset =
                                    # KgForestDataset('train_320.txt',
                                    transform=[
                                        # tif_color_corr,
                                        augment,
                                        img_to_tensor,
                                    ],
                                    outfields=['tif', 'jpg', 'label'],
                                    height=SIZE, width=SIZE,
                                    )

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        # drop_last   = True,
        num_workers=2,
        pin_memory=True)

    test_dataset = KgForestDataset('val_5000.txt',
                                   # test_dataset =
                                   # KgForestDataset('val_320.txt',
                                   height=SIZE, width=SIZE,
                                   transform=[
                                       # tif_color_corr,
                                       img_to_tensor,
                                   ],
                                   outfields=['tif', 'jpg', 'label'],
                                   cacheGB=6,
                                   )

    # num worker = 0 is important
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),  # None,
        batch_size=batch_size,
        # drop_last   = False,
        num_workers=0,
        pin_memory=True)

    print('\tbatch_size           = %d\n' % batch_size)
    print('\ttrain_loader.sampler = %s\n' % (str(train_loader.sampler)))
    print('\ttest_loader.sampler  = %s\n' % (str(test_loader.sampler)))
    print('\n')

    net, optimizer, start_epoch = get_model(init_file_list)
    if use_gpu:
        net.cuda()

    # optimiser ----------------------------------
    # LR = StepLR([ (0,0.1),  (10,0.01),  (25,0.005),  (35,0.001), (40,0.0001), (43,-1)])
    # fine tunning
    LR = StepLR([(0, 0.01),  (10, 0.005),
                 (23, 0.001),  (35, 0.0001), (38, -1)])
    #LR = CyclicLR(base_lr=0.001, max_lr=0.01, step=5., mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle')

    num_epoches = 80  # 100
    it_print = 20  # 20
    epoch_test = 1
    epoch_save = 5

    # start training here! ##############################################3
    print('** start training here! **\n')

    print(' optimizer=%s\n' % str(optimizer))
    print(' LR=%s\n\n' % str(LR))
    print(' epoch   iter   rate  |  smooth_loss   |  train_loss  (acc)  |  valid_loss  (acc)  | min\n')
    print('----------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    train_loss = np.nan
    train_acc = np.nan
    test_loss = np.nan
    test_acc = np.nan
    best_acc = 0
    time = 0

    start0 = timer()
    for epoch in range(start_epoch, num_epoches):  # loop over the dataset multiple times
        start = timer()

        #---learning rate schduler ------------------------------
        lr = LR.get_rate(epoch, num_epoches)
        if lr < 0:
            break

        adjust_learning_rate(optimizer, lr)
        #--------------------------------------------------------

        smooth_loss_sum = 0.0
        smooth_loss_n = 0

        if use_gpu:
            net.cuda().train()
        else:
            net.train()
        num_its = len(train_loader)
        for it, batch in enumerate(train_loader, 0):
            # images = batch['tif'][:,1:,:,:] #IR R G B to R G B
            imagesRGB = batch['jpg']
            imagesNIR = batch['tif'][:, 0:1, :, :]
            if use_gpu:
                imagesRGB = imagesRGB.cuda()
                imagesNIR = imagesNIR.cuda()

            labels = batch['label'].float()

            output_cat, probs = net(Variable(imagesNIR), Variable(imagesRGB))
            if use_gpu:
                loss = loss_func(output_cat, labels.cuda())
            else:
                loss = loss_func(output_cat, labels)

            #logits, probs = net(Variable(images.cuda()))
            #loss  = loss_func(logits, labels.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # additional metrics
            smooth_loss_sum += loss.data[0]
            smooth_loss_n += 1

            # print statistics
            if it % it_print == it_print - 1:
                smooth_loss = smooth_loss_sum / smooth_loss_n
                smooth_loss_sum = 0.0
                smooth_loss_n = 0

                if use_gpu:
                    train_acc = multi_f_measure(probs.data, labels.cuda())
                else:
                    train_acc = multi_f_measure(probs.data, labels)
                train_loss = loss.data[0]

                print('\r%5.1f   %5d    %0.4f   |  %0.4f  | %0.4f  %6.4f | ... ' %
                      (epoch + it / num_its, it + 1, lr,
                       smooth_loss, train_loss, train_acc),
                      )
            # modified by steve
            # end='',flush=True

        #---- end of one epoch -----
        end = timer()
        time = (end - start) / 60

        if epoch % epoch_test == epoch_test - 1 or epoch == num_epoches - 1:
            if use_gpu:
                net.cuda().eval()
            else:
                net.eval()
            test_logits, test_probs = do_predict(
                net, test_dataset)
            test_labels = torch.from_numpy(
                test_dataset.df[CLASS_NAMES].values.astype(np.float32))
            test_acc = f2_score(test_probs, test_labels.numpy())
            test_loss = loss_func(torch.autograd.Variable(
                torch.from_numpy(test_logits)), test_labels).data[0]
            # modified by steve
            # print('\r',end='',flush=True)
            print('\r')
            print('%5.1f   %5d    %0.4f   |  %0.4f  | %0.4f  %6.4f | %0.4f  %6.4f  |  %3.1f min \n' %
                  (epoch + 1, it + 1, lr, smooth_loss, train_loss, train_acc, test_loss, test_acc, time))

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(net, out_dir + '/snap/mixed_cat_best.torch')

    #---- end of all epoches -----
    end0 = timer()
    time0 = (end0 - start0) / 60

    # check : load model and re-test
    net = torch.load(out_dir + '/snap/mixed_cat_best.torch')
    if use_gpu:
        net.cuda().eval()
    else:
        net.eval()
    test_logits, test_probs = do_predict(
        net, test_dataset)
    test_labels = torch.from_numpy(
        test_dataset.df[CLASS_NAMES].values.astype(np.float32))
    test_acc = f2_score(test_probs, test_labels.numpy())
    test_loss = loss_func(torch.autograd.Variable(
        torch.from_numpy(test_logits)), test_labels).data[0]

    print('\n')
    print('%s:\n' % (out_dir + '/snap/mixed_cat_best.torch'))
    print('\tall time to train=%0.1f min\n' % (time0))
    print('\ttest_loss=%f, test_acc=%f\n' % (test_loss, test_acc))

    return net

if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    do_training(out_dir='../../output/inception_and_resnet')

    # find thres
    net, _, _ = get_model(
        "../../output/inception_and_resnet/snap/mixed_cat_best.torch")
    train_dataset = KgForestDataset('labeled.txt',
                                    transform=[
                                        # tif_color_corr,
                                        img_to_tensor,
                                    ],
                                    outfields=['tif', 'jpg', 'label'],
                                    height=SIZE, width=SIZE,
                                    )
    logits, probs = do_predict(net, train_dataset, silent=False)
    labels = train_dataset.df[CLASS_NAMES].values.astype(np.float32)
    best_single_thres, best_all_thres = do_thresholds(probs, labels)

    # do submit
    # net, _, _ = get_model(
    #     "../../output/inception_and_resnet/snap/mixed_cat_best.torch")
    test_dataset = KgForestDataset('unlabeled.txt',  # 'unlabeled.txt',
                                   transform=[
                                       # tif_color_corr,
                                       img_to_tensor,
                                   ],
                                   outfields=['tif', 'jpg'],
                                   height=SIZE, width=SIZE,
                                   )
    logits, probs = do_predict(net, test_dataset, silent=False)

    # from resnet34_tif_rgb
    ###best_threshold = np.ones(len(CLASS_NAMES))* 0.2200
    # best_thresholds = np.array( [ 0.170, 0.245, 0.150, 0.195, 0.145, 0.230, 0.225, 0.245, 0.190, 0.240,
    # 0.095, 0.305, 0.255, 0.135, 0.145, 0.240, 0.060] )

    # from resnet34_tif_irgb
    ###best_threshold = np.ones(len(CLASS_NAMES))* 0.2250
    # best_thresholds = np.array([ 0.190, 0.250, 0.225, 0.125, 0.270, 0.235, 0.200, 0.240, 0.240, 0.250,
    # 0.120, 0.100, 0.240, 0.150, 0.210, 0.190, 0.050])

    # from resnet34_tif_irgb
    ##best_threshold = np.ones(len(CLASS_NAMES))* 0.2200
    # best_thresholds = np.array(
    # [ 0.195, 0.235, 0.230, 0.095, 0.295, 0.215, 0.175, 0.250, 0.220, 0.250,
    # 0.130, 0.345, 0.145, 0.170, 0.215, 0.260, 0.090]
    # )

    # from resnet34_tif_irrgb
    #best_threshold = np.ones(len(CLASS_NAMES))* 0.2100
    # best_thresholds = np.array(
    #        [ 0.155, 0.260, 0.225, 0.110, 0.280, 0.240, 0.225, 0.205, 0.205, 0.250,
    #              0.080, 0.145, 0.180, 0.075, 0.130, 0.160, 0.075]
    #                             )
    do_submit(probs, best_all_thres, test_dataset.df.index,  "submit_mixed_cat.csv")
