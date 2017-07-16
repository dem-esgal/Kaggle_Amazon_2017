
from dataset.kgforest import *
from dataset.tool import *

from net.rates import *
from net.util import *

from net.model.resnet import resnet34 as Net
from net.model.inceptionv3 import Inception3 as Net_Inception

import torch.nn.functional as F
import os.path

SIZE =  256 #288   #256   #128  #112
SRC = 'tif'  
#SRC = 'jpg' #channel
CH = 'rgb'
#CH = 'irrg'
# CH = 'irrgb'
SEED = 123

def loss_func(logits, labels):
    loss = nn.MultiLabelSoftMarginLoss()(logits, Variable(labels))
    return loss

def multi_f_measure( probs, labels, threshold=0.235, beta=2 ):

    SMALL = 1e-6 #0  #1e-12
    batch_size = probs.size()[0]

    #weather
    l = labels
    p = (probs>threshold).float()

    num_pos     = torch.sum(p,  1)
    num_pos_hat = torch.sum(l,  1)
    tp          = torch.sum(l*p,1)
    precise     = tp/(num_pos     + SMALL)
    recall      = tp/(num_pos_hat + SMALL)

    fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
    f  = fs.sum()/batch_size
    return f

def f2_score(y_pred, y_true, thres=0.235):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    y_pred = y_pred>thres
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

def augment(x, u=0.75):
    if random.random()<u:
        if random.random()>0.5:
            x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
        else:
            x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)
        x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, u=1)

    x = randomFlip(x, u=0.5)
    x = randomTranspose(x, u=0.5)
    #x = randomContrast(x, limit=0.2, u=0.5)
    #x = randomSaturation(x, limit=0.2, u=0.5),
    x = randomFilter(x, limit=0.5, u=0.2)
    return x

def do_thresholds(probs,labels):
    print('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    print('\n')

    nClass = len(CLASS_NAMES)
    tryVals = np.arange(0,1,0.005)
    scores = np.zeros(len(tryVals))

    #single threshold
    for i,t in enumerate(tryVals):
        scores[i] = fbeta_score(labels, probs>t, beta=2, average='samples')

    best_single_thres = tryVals[scores.argmax()]
    best_single_thres_score = scores.max()

    print('*best_threshold (fixed)*\n')
    print("%0.4f"%best_single_thres)
    print('\n')
    print('*best_score*\n')
    print('%0.4f\n'%best_single_thres_score)

    #per class threshold
    best_thresholds = np.ones(nClass) * best_single_thres
    best_multi_thres_score = 0
    noChange = 0
    for iter in range(nClass*10):
        print ("thres scan iter %i"%iter)
        trial_thresholds = best_thresholds.copy()
        targetClass = iter%nClass
        for i,t in enumerate(tryVals):
            trial_thresholds[targetClass] = t
            scores[i] = fbeta_score(labels, probs > trial_thresholds, beta=2, average='samples')

        best_threshold = tryVals[scores.argmax()]
        best_multi_thres_score = scores.max()
        if best_threshold==best_thresholds[targetClass]:
            noChange+=1
        else:
            noChange=0
            best_thresholds[targetClass] = best_threshold

        if noChange==nClass: break

    print('*best_threshold (per class)*\n')
    print(np.array2string(best_thresholds, formatter={'float_kind':lambda x: ' %.3f' % x}, separator=','))
    print('\n')
    print('*best_score*\n')
    print('%0.4f\n'%best_multi_thres_score)

    return best_single_thres, best_thresholds

def do_predict(net, net_Inception, fc_cat, fc_v2, dataset, batch_size=20, silent=True):
    net.cuda().eval()

    num_classes  = len(CLASS_NAMES)
    logits  = np.zeros((len(dataset),num_classes),np.float32)
    probs  = np.zeros((len(dataset),num_classes),np.float32)

    tot_samples = 0

    loader  = DataLoader(
                        dataset,
                        sampler     = SequentialSampler(dataset),  #None,
                        batch_size  = batch_size,
                        # drop_last   = False,
                        num_workers = 0,
                        pin_memory  = True)

    for it, batch in enumerate(loader, 0):
        if not silent: print("predict batch %i / %i"%(it, len(loader)))
        #images = batch['tif'][:,1:,:,:] #IR R G B to R G B
        images = batch[SRC]
        imagesNIR = batch[SRC][:,0:1,:,:]
        if SRC=='tif':
            if CH == 'rgb'  : images = images[:,1:,:,:] #IR R G B to  R G B
            if CH == 'irrg' : images = images[:,:3,:,:] #IR R G B to IR R G
            if CH == 'irrgb': pass

        batch_size = len(images)
        tot_samples += batch_size
        start = tot_samples-batch_size
        end   = tot_samples

        # forward
        #ls, ps = net(Variable(images.cuda(),volatile=True))
        vector1 = net(Variable(images.cuda(),volatile=True))
        vector2 = net_Inception(Variable(imagesNIR.cuda(),volatile=True))
        vector2 = fc_v2(vector2)
        vector_cat = torch.cat([vector1,vector2],1).cuda()
        #modified by steve
        output_cat = fc_cat(vector_cat).cuda()
        probs_cat = F.sigmoid(output_cat)
        #print(probs.data.cpu().numpy().reshape(-1,num_classes))
        logits[start:end] = output_cat.data.cpu().numpy().reshape(-1,num_classes)
        probs[start:end]  = probs_cat.data.cpu().numpy().reshape(-1,num_classes)

    assert(len(dataset)==tot_samples)

    return logits,probs

def do_submit(prob, thres, imgKeys, outfile = "submit.csv"):
    tagsVec = probs>thres
    tagsCol = []
    for arow in tagsVec:
        tags = [CLASS_NAMES[i] for i in np.where(arow)[0] ]
        tags = " ".join(tags)
        tagsCol.append(tags)
    output = pd.DataFrame()
    output['image_name'] = imgKeys
    output['tags'] = tagsCol
    output.to_csv(outfile, index=False)


def get_model(init_file=None):
    print('** net setting **\n')
    num_classes = len(CLASS_NAMES)
    net = Net(in_shape = (3, SIZE, SIZE), num_classes=num_classes)
    net_Inception = Net_Inception(in_shape = (1, SIZE, SIZE), num_classes=num_classes)
    print('%s\n\n ResNet Type:'%(type(net)))
    print('%s\n\n Inception Type:'%(type(net_Inception)))
    print('===== Combined models ======')
    print('\n')

    #for param in net.parameters():
    #        param.requires_grad = False
    #untrainable_layers = [
    #    net.conv1,
    #    #net.bn1,
    #    net.relu,
    #    net.maxpool,
    #    net.layer1,
    #    net.layer2,
    #    net.layer3
    #]
    #for aLayer in untrainable_layers:
    #    for param in aLayer.parameters():
    #        param.requires_grad = False

    #trainable_layers = [net.layer4, net.fc]
    #trainable_paras = []
    #for aLayer in trainable_layers:
    #    for param in aLayer.parameters():
    #        param.requires_grad = True
    #        trainable_paras.append(param)
    train_param = []
    for param in net.parameters():
    	param.requires_grad = True
    	train_param.append(param)
    for param in net_Inception.parameters():
    	param.requires_grad = True
    	train_param.append(param)

    #optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    optimizer = optim.SGD(train_param, lr=0.1, momentum=0.9, weight_decay=0.0005)  ###0.0005

    #optimizer = optim.SGD(trainable_paras, lr=0.1, momentum=0.9, weight_decay=0.0005)  ###0.0005

    ## resume from previous ----------------------------------
    start_epoch=0
    if init_file is not None:
        init_content = torch.load(init_file)
        init_content_inception = torch.load('../../input/inception_v3_google-1a9a5a14.pth')
        if isinstance(init_content, dict):
            if 'epoch' in init_content:
                # checkpoint
                start_epoch = init_content['epoch']
                optimizer.load_state_dict( init_content['optimizer_state'])
                load_model_weight(net, init_content['model_state'],skip_list=[])
            else:
                # pretrained weights for inception model
                skip_list = ['Conv2d_1a_3x3.conv.weight','fc.weight', 'fc.bias']
                load_model_weight(net_Inception, init_content_inception, skip_list=skip_list)
                tmp = torch.mean(init_content_inception['Conv2d_1a_3x3.conv.weight'],1) 
                if isinstance(tmp, torch.nn.Parameter): tmp = tmp.data
                net_Inception.state_dict()['Conv2d_1a_3x3.conv.weight'][:,0,:,:] = tmp

                # pretrained weights for ResNet
                if CH=='irrgb':
                    skip_list = ['conv1.weight','fc.weight', 'fc.bias']
                    load_model_weight(net, init_content, skip_list=skip_list)
                    tmp = init_content['conv1.weight']
                    if isinstance(tmp, torch.nn.Parameter): tmp = tmp.data
                    net.state_dict()['conv1.weight'][:,1:,:,:] = tmp
                else:
                    skip_list = ['fc.weight', 'fc.bias']
                    load_model_weight(net, init_content, skip_list=skip_list)
        else:
            # full model
            net = init_content

    return net, net_Inception, optimizer, start_epoch

def do_training(out_dir='../../output/inception_and_resnet'):

    init_file = '../../input/resnet34-333f7ec4.pth'

    ## ------------------------------------
    if not os.path.exists(out_dir +'/snap'):
        os.makedirs(out_dir +'/snap')
    if not os.path.exists(out_dir +'/checkpoint'):
        os.makedirs(out_dir +'/checkpoint')

    print('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    print('** some experiment setting **\n')
    print('\tSEED    = %u\n' % SEED)
    print('\tfile    = %s\n' % __file__)
    print('\tout_dir = %s\n' % out_dir)
    print('\n')

    ## dataset ----------------------------------------
    print('** dataset setting **\n')
    num_classes = len(CLASS_NAMES)
    batch_size  = 20 #48  #96 #96  #80 #96 #96   #96 #32  #96 #128 #

    train_dataset = KgForestDataset('train_35479.txt',
    #train_dataset = KgForestDataset('train_320.txt',
                                    transform=[
                                        #tif_color_corr,
                                        augment,
                                        img_to_tensor,
                                        ],
                                    outfields = [SRC, 'label'],
                                    height=SIZE, width=SIZE,
                                   )

    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),  ##
                        batch_size  = batch_size,
                        # drop_last   = True,
                        num_workers = 2,
                        pin_memory  = True)

    test_dataset = KgForestDataset('val_5000.txt',
    #test_dataset = KgForestDataset('val_320.txt',
                                height=SIZE, width=SIZE,
                                transform=[
                                    #tif_color_corr,
                                    img_to_tensor,
                                ],
                                outfields = [SRC,'label'],
                                cacheGB=6,
                              )

    #num worker = 0 is important
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),  #None,
                        batch_size  = batch_size,
                        # drop_last   = False,
                        num_workers = 0,
                        pin_memory  = True)

    print('\tbatch_size           = %d\n'%batch_size)
    print('\ttrain_loader.sampler = %s\n'%(str(train_loader.sampler)))
    print('\ttest_loader.sampler  = %s\n'%(str(test_loader.sampler)))
    print('\n')

    net,net_Inception,optimizer,start_epoch = get_model(init_file)
    net.cuda()

    # optimiser ----------------------------------
    # LR = StepLR([ (0,0.1),  (10,0.01),  (25,0.005),  (35,0.001), (40,0.0001), (43,-1)])
    # fine tunning
    LR = StepLR([ (0, 0.01),  (10, 0.005),  (23, 0.001),  (35, 0.0001), (38,-1)])
    #LR = CyclicLR(base_lr=0.001, max_lr=0.01, step=5., mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle')

    num_epoches = 50  #100
    it_print    = 20  #20
    epoch_test  = 1
    epoch_save  = 5


    ## start training here! ##############################################3
    print('** start training here! **\n')

    print(' optimizer=%s\n'%str(optimizer) )
    print(' LR=%s\n\n'%str(LR) )
    print(' epoch   iter   rate  |  smooth_loss   |  train_loss  (acc)  |  valid_loss  (acc)  | min\n')
    print('----------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    train_loss  = np.nan
    train_acc   = np.nan
    test_loss   = np.nan
    test_acc    = np.nan
    best_acc   = 0
    time = 0

    start0 = timer()
    for epoch in range(start_epoch, num_epoches):  # loop over the dataset multiple times
        start = timer()

        #---learning rate schduler ------------------------------
        lr =  LR.get_rate(epoch, num_epoches)
        if lr<0 :break

        adjust_learning_rate(optimizer, lr)
        #--------------------------------------------------------

        smooth_loss_sum = 0.0
        smooth_loss_n   = 0
        
        net.cuda().train()
        net_Inception.cuda().train()
        num_its = len(train_loader)
        for it, batch in enumerate(train_loader, 0):
            #images = batch['tif'][:,1:,:,:] #IR R G B to R G B
            images = batch[SRC]
            imagesNIR = batch[SRC][:,0:1,:,:]
            if SRC=='tif':
                if CH == 'rgb'  : images = images[:,1:,:,:] #IR R G B to  R G B
                if CH == 'irrg' : images = images[:,:3,:,:] #IR R G B to IR R G
                if CH == 'irrgb': pass
            labels = batch['label'].float()
            vector1 = net(Variable(images.cuda()))
            vector2 = net_Inception(Variable(imagesNIR.cuda()))
            # modified by steve (simple dimension reduction)
            fc_v2 = torch.nn.Linear(vector2.size(1),vector1.size(1)//3).cuda()
            vector2 = fc_v2(vector2).cuda()
            vector_cat = torch.cat([vector1,vector2],1).cuda()
            fc_cat = torch.nn.Linear(vector_cat.size(1),17).cuda()
            output_cat = fc_cat(vector_cat).cuda()
            probs = F.sigmoid(output_cat)
            loss  = loss_func(output_cat, labels.cuda())

            #logits, probs = net(Variable(images.cuda()))
            #loss  = loss_func(logits, labels.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #additional metrics
            smooth_loss_sum += loss.data[0]
            smooth_loss_n   += 1

            # print statistics
            if it % it_print == it_print-1:
                smooth_loss = smooth_loss_sum/smooth_loss_n
                smooth_loss_sum = 0.0
                smooth_loss_n = 0

                train_acc  = multi_f_measure(probs.data, labels.cuda())
                train_loss = loss.data[0]

                print('\r%5.1f   %5d    %0.4f   |  %0.4f  | %0.4f  %6.4f | ... ' % \
                        (epoch + it/num_its, it + 1, lr, smooth_loss, train_loss, train_acc),\
                        )
            # modified by steve
            #end='',flush=True



        #---- end of one epoch -----
        end = timer()
        time = (end - start)/60

        if epoch % epoch_test == epoch_test-1  or epoch == num_epoches-1:
            net.cuda().eval()
            test_logits,test_probs = do_predict(net, net_Inception, fc_cat, fc_v2, test_dataset)
            test_labels = torch.from_numpy(test_dataset.df[CLASS_NAMES].values.astype(np.float32))
            test_acc  = f2_score(test_probs, test_labels.numpy())
            test_loss = loss_func(torch.autograd.Variable(torch.from_numpy(test_logits)), test_labels).data[0]
            # modified by steve
            #print('\r',end='',flush=True)
            print('\r')
            print('%5.1f   %5d    %0.4f   |  %0.4f  | %0.4f  %6.4f | %0.4f  %6.4f  |  %3.1f min \n' % \
                    (epoch + 1, it + 1, lr, smooth_loss, train_loss, train_acc, test_loss,test_acc, time))

            if test_acc>best_acc:
                best_acc = test_acc
                torch.save(net, out_dir +'/snap/best_acc_%s_%03d.torch'%(("%.4f"%best_acc).replace('.','d'), epoch+1))
                torch.save(net_Inception, out_dir +'/snap/best_acc_inception_%s_%03d.torch'%(("%.4f"%best_acc).replace('.','d'), epoch+1))
                # modified by steve
                torch.save(fc_cat, out_dir +'/snap/best_acc_fc_cat_%s_%03d.torch'%(("%.4f"%best_acc).replace('.','d'), epoch+1))
                # modified by steve
                torch.save(fc_v2, out_dir +'/snap/best_acc_fc_v2_%s_%03d.torch'%(("%.4f"%best_acc).replace('.','d'), epoch+1))

        if epoch % epoch_save == epoch_save-1 or epoch == num_epoches-1:
            #torch.save(net, out_dir +'/snap/%03d.torch'%(epoch+1))
            torch.save({
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch'     : epoch,
            }, out_dir +'/checkpoint/%03d.pth'%(epoch+1))
            ## https://github.com/pytorch/examples/blob/master/imagenet/main.py


    #---- end of all epoches -----
    end0  = timer()
    time0 = (end0 - start0) / 60

    ## check : load model and re-test
    torch.save(net,out_dir +'/snap/final.torch')

    net = torch.load(out_dir +'/snap/final.torch')
    net.cuda().eval()
    test_logits,test_probs = do_predict(net, net_Inception, fc_cat, fc_v2, test_dataset)
    test_labels = torch.from_numpy(test_dataset.df[CLASS_NAMES].values.astype(np.float32))
    test_acc  = f2_score(test_probs, test_labels.numpy())
    test_loss = loss_func(torch.autograd.Variable(torch.from_numpy(test_logits)), test_labels).data[0]

    print('\n')
    print('%s:\n'%(out_dir +'/snap/final.torch'))
    print('\tall time to train=%0.1f min\n'%(time0))
    print('\ttest_loss=%f, test_acc=%f\n'%(test_loss,test_acc))

    return net

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    do_training(out_dir='../../output/inception_and_resnet')

    ## find thres
    #net,_,_ = get_model("../../output/resnet34_tif_irrgb_nocorr_out/snap/best_acc_0d9221_026.torch")
    #train_dataset = KgForestDataset('labeled.txt',
    #                                transform=[
    #                                    #tif_color_corr,
    #                                    img_to_tensor,
    #                                    ],
    #                                outfields = [SRC, 'label'],
    #                                height=SIZE, width=SIZE,
    #                               )
    #logits, probs = do_predict(net, train_dataset, silent=False)
    #labels = train_dataset.df[CLASS_NAMES].values.astype(np.float32)
    #do_thresholds(probs, labels)


    ## do submit
    #net,_,_ = get_model("../../output/resnet34_tif_irrgb_nocorr_out/snap/best_acc_0d9221_026.torch")
    #test_dataset = KgForestDataset('unlabeled.txt',
    #                                transform=[
    #                                    #tif_color_corr,
    #                                    img_to_tensor,
    #                                    ],
    #                                outfields = [SRC],
    #                                height=SIZE, width=SIZE,
    #                               )
    #logits, probs = do_predict(net, test_dataset, silent=False)

    ###from resnet34_tif_rgb
    ###best_threshold = np.ones(len(CLASS_NAMES))* 0.2200
    ###best_thresholds = np.array( [ 0.170, 0.245, 0.150, 0.195, 0.145, 0.230, 0.225, 0.245, 0.190, 0.240,
    ###                              0.095, 0.305, 0.255, 0.135, 0.145, 0.240, 0.060] )

    ###from resnet34_tif_irgb
    ###best_threshold = np.ones(len(CLASS_NAMES))* 0.2250    
    ###best_thresholds = np.array([ 0.190, 0.250, 0.225, 0.125, 0.270, 0.235, 0.200, 0.240, 0.240, 0.250,
    ###                             0.120, 0.100, 0.240, 0.150, 0.210, 0.190, 0.050])

    ###from resnet34_tif_irgb
    ##best_threshold = np.ones(len(CLASS_NAMES))* 0.2200
    ##best_thresholds = np.array(
    ##        [ 0.195, 0.235, 0.230, 0.095, 0.295, 0.215, 0.175, 0.250, 0.220, 0.250,
    ##          0.130, 0.345, 0.145, 0.170, 0.215, 0.260, 0.090]
    ##                             )

    ##from resnet34_tif_irrgb
    #best_threshold = np.ones(len(CLASS_NAMES))* 0.2100
    #best_thresholds = np.array(
    #        [ 0.155, 0.260, 0.225, 0.110, 0.280, 0.240, 0.225, 0.205, 0.205, 0.250,
    #              0.080, 0.145, 0.180, 0.075, 0.130, 0.160, 0.075]
    #                             )
    #do_submit(probs, best_thresholds, test_dataset.df.index,  "submit_resnet34_tif_irrgb.csv")

