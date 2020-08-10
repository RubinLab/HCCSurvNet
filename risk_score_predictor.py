import os
import glob
import time
from PIL import Image
import copy
import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length, check_array
from sksurv.metrics import concordance_index_ipcw

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torchvision import transforms
from utils import new_transforms

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):

        self.df = df
        self.transform = transform

    def __getitem__(self, index):
        path = self.df.path[index]
        pt = self.df.pt[index]
        img = np.array(Image.open(path))
        event = self.df.event[index]
        time = self.df.time[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, event, time, pt

    def __len__(self):
        return len(self.df)

augment = transforms.Compose([transforms.ToPILImage(),
                              new_transforms.Resize((imgSize, imgSize)),
                              transforms.RandomHorizontalFlip(),
                              new_transforms.RandomRotate(),
                              new_transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform = transforms.Compose([transforms.ToPILImage(),
                                new_transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def train_model(model, loaders, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_cindex = 0.0
    best_loss = float("inf")
    counter = 0
    state_dicts = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_numerator = 0.0
            running_denominator = 0.0

            for inputs, events, times, pt in loaders[phase]:
                inputs = inputs.to(device)
                events = events.to(device)
                times = times.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = neg_partial_log_likelihood(outputs, events, times)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, _, _, _, _, _numerator, _denominator = concordance_index_censored(events, times, outputs, tied_tol=1e-8)
                running_numerator += _numerator
                running_denominator += _denominator

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_cindex = running_numerator / running_denominator

            print('{} Loss: {:.4f} C-index: {:.4f}'.format(phase, epoch_loss, epoch_cindex))

            if phase == 'val':
                state_dicts.append(copy.deepcopy(model.state_dict()))
            
            if phase == 'val':
                if scheduler is not None:
                    scheduler.step(epoch_loss)
            
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    best_cindex = epoch_cindex
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1
        print()
        
        if counter > 4:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val C-index: {:4f}'.format(best_cindex))
    print('Best val Loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model, state_dicts, best_loss, best_cindex, best_epoch

def test_model(model, loader, dataset_size):
    
    print('-' * 10)
    model.eval()
    running_loss = 0.0
    running_numerator = 0.0
    running_denominator = 0.0
    whole_outputs = torch.FloatTensor(dataset_size)
    whole_events = torch.LongTensor(dataset_size)
    whole_times = torch.LongTensor(dataset_size)
    pts = []
    
    with torch.no_grad():

        for i, data in enumerate(loader):
            inputs = data[0].to(device)
            events = data[1].to(device)
            times = data[2].to(device)
            pt = data[3]

            outputs = model(inputs)
            loss = neg_partial_log_likelihood(outputs, events, times)

            running_loss += loss.item() * inputs.size(0)

            whole_outputs[i*batchSize:i*batchSize+inputs.size(0)]=outputs.detach().squeeze().clone()
            whole_events[i*batchSize:i*batchSize+inputs.size(0)]=events.detach().clone()
            whole_times[i*batchSize:i*batchSize+inputs.size(0)]=times.detach().clone()
            for p in pt:
                pts.append(p)

        total_loss = running_loss / dataset_size

    print('Test Loss: {:.4f}'.format(total_loss))

    return whole_outputs.cpu().numpy(), whole_events.cpu().numpy(), whole_times.cpu().numpy(), total_loss, pts

def R_set(x):
    """
    Based on https://github.com/tomcat123a/survival_loss_criteria/blob/master/loss_function_criteria.py
    """
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)

    return indicator_matrix


def neg_partial_log_likelihood(pred, yevent, ytime):
    """
    Based on https://github.com/tomcat123a/survival_loss_criteria/blob/master/loss_function_criteria.py
    """
    ytime_sorted, idx = torch.sort(ytime, dim = -1, descending=True)
    yevent_sorted = torch.gather(yevent, -1, idx)
    pred_sorted = torch.gather(pred.view(-1), -1, idx)
    pred_sorted = pred_sorted.view(-1, 1)
    n_observed = int(yevent_sorted.sum(0))
    ytime_indicator = R_set(ytime_sorted)
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()
    risk_set_sum = ytime_indicator.mm(torch.exp(pred_sorted)) 
    diff = pred_sorted - torch.log(risk_set_sum)
    yevent_sorted = yevent_sorted.float()
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent_sorted.view(-1,1))
    loss = (- (sum_diff_in_observed / n_observed)).reshape((-1,))

    return loss

def _check_estimate(estimate, test_time):
    """
    Based on https://github.com/sebp/scikit-survival
    """
    estimate = check_array(estimate, ensure_2d=False)
    if estimate.ndim != 1:
        raise ValueError(
            'Expected 1D array, got {:d}D array instead:\narray={}.\n'.format(
                estimate.ndim, estimate))
    check_consistent_length(test_time, estimate)
    return estimate

def _check_inputs(event_indicator, event_time, estimate):
    """
    Based on https://github.com/sebp/scikit-survival
    """
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False)
    event_time = check_array(event_time, ensure_2d=False)
    estimate = _check_estimate(estimate, event_time)

    if not np.issubdtype(event_indicator.dtype, np.bool_):
        raise ValueError(
            'only boolean arrays are supported as class labels for survival analysis, got {0}'.format(
                event_indicator.dtype))

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate

def _get_comparable(event_indicator, event_time):
    """
    Based on https://github.com/sebp/scikit-survival
    """
    order = np.argsort(event_time)
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = np.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time

def _estimate_concordance_index(event_indicator, event_time, estimate, tied_tol=1e-8):
    """
    Based on https://github.com/sebp/scikit-survival
    """
    weights = np.ones_like(estimate)
    order = np.argsort(event_time)
    comparable, tied_time = _get_comparable(event_indicator, event_time)

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        ties = np.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    if denominator == 0:
        cindex = np.inf
    else:
        cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time, numerator, denominator

def concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-8):
    """
    Based on https://github.com/sebp/scikit-survival

    Concordance index for right-censored data
    The concordance index is defined as the proportion of all comparable pairs
    in which the predictions and outcomes are concordant.
    Samples are comparable if for at least one of them an event occurred.
    If the estimated risk is larger for the sample with a higher time of
    event/censoring, the predictions of that pair are said to be concordant.
    If an event occurred for one sample and the other is known to be
    event-free at least until the time of event of the first, the second
    sample is assumed to *outlive* the first.
    When predicted risks are identical for a pair, 0.5 rather than 1 is added
    to the count of concordant pairs.
    A pair is not comparable if an event occurred for both of them at the same
    time or an event occurred for one of them but the time of censoring is
    smaller than the time of event of the first one.
    See [1]_ for further description.
    Parameters
    ----------
    event_indicator : array-like, shape = (n_samples,)
        Boolean array denotes whether an event occurred --> can take torch.tensor with 0 and 1
    event_time : array-like, shape = (n_samples,)
        Array containing the time of an event or time of censoring --> can take torch.tensor
    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event --> can take torch.tensor
    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.
    Returns
    -------
    cindex : float
        Concordance index
    concordant : int
        Number of concordant pairs
    discordant : int
        Number of discordant pairs
    tied_risk : int
        Number of pairs having tied estimated risks
    tied_time : int
        Number of comparable pairs sharing the same time
    numerator : int
    denominator : int
    
    References
    ----------
    .. [1] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A,
           "Multivariable prognostic models: issues in developing models,
           evaluating assumptions and adequacy, and measuring and reducing errors",
           Statistics in Medicine, 15(4), 361-87, 1996.
    """
    
    event_indicator = np.array([bool(i) for i in event_indicator.cpu().numpy()])
    event_time = event_time.cpu().numpy()
    estimate = estimate.cpu().detach().view(-1).numpy()
    event_indicator, event_time, estimate = _check_inputs(
        event_indicator, event_time, estimate)

    return _estimate_concordance_index(event_indicator, event_time, estimate, tied_tol)

def aggoutputs(outputs, method):
    if method=='mean':
        agg = np.mean(outputs)
    elif method=='max':
        agg = np.max(outputs)
    elif method=='90percentile':
        agg = np.percentile(outputs, 90)
    elif method=='80percentile':
        agg = np.percentile(outputs, 80)
    elif method=='70percentile':
        agg = np.percentile(outputs, 70)
    elif method=='60percentile':
        agg = np.percentile(outputs, 60)
    elif method=='median':
        agg = np.median(outputs)
    else:
        raise Excerption('Not supported method')
    return agg

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    np.random.seed(123456)
    _ = torch.manual_seed(123456)

    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv = 'path/to/tcga_metadata.csv' # please download from https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018
    df = pd.read_csv(csv)

    imgs = glob.glob('/path/to/top100tiles/*.png')

    fnames = [os.path.basename(i) for i in imgs]
    slides = [i[:23] for i in fnames]
    pts = [i[:12] for i in slides]

    events = []
    times = []
    for i in pts:
        ev = df[df.bcr_patient_barcode==i]['PFI'].values[0]
        ti = df[df.bcr_patient_barcode==i]['PFI.time'].values[0]
        events.append(ev)
        times.append(ti)

    df = pd.DataFrame(columns=['path', 'fname', 'slide', 'pt', 'event', 'time'])
    df.path = img
    df.fname = fnames
    df.slide = slides
    df.pt = pts
    df.event = events
    df.time = times

    df_dict = {}
    for name, group in df.groupby('pt'):
        df_dict[name] = group

    pts = list(df_dict.keys())

    dict_pts = {}
    rand = np.arange(len(df_dict))
    np.random.shuffle(rand)
    n = 0
    test_ids = rand[n*54:(n+1)*54]
    dict_pts['test'] = np.array(pts)[test_ids]
    val_ids = rand[(n+1)*54:(n+2)*54]
    dict_pts['val'] = np.array(pts)[val_ids]
    train_ids = np.array(list(set(rand.tolist())-set(test_ids)-set(val_ids)))
    dict_pts['train'] = np.array(pts)[train_ids]

    train_df = pd.DataFrame(columns=df.columns)
    for i in dict_pts['train']:
        train_df = pd.concat([train_df, df_dict[i]])
    train_df = train_df.reset_index(drop=True)
        
    val_df = pd.DataFrame(columns=df.columns)
    for i in dict_pts['val']:
        val_df = pd.concat([val_df, df_dict[i]])
    val_df = val_df.reset_index(drop=True)
        
    test_df = pd.DataFrame(columns=df.columns)
    for i in dict_pts['test']:
        test_df = pd.concat([test_df, df_dict[i]])
    test_df = test_df.reset_index(drop=True)

    datasets = {}
    loaders = {}
    for dset_type in ['train', 'val', 'test']:
        if dset_type == 'train':
            datasets[dset_type] = MyDataset(train_df, transform = augment)
            loaders[dset_type] = torch.utils.data.DataLoader(datasets[dset_type], batch_size=batchSize, shuffle=True)
        elif dset_type == 'val':
            datasets[dset_type] = MyDataset(val_df, transform = transform)
            loaders[dset_type] = torch.utils.data.DataLoader(datasets[dset_type], batch_size=batchSize, shuffle=True)
        elif dset_type == 'test':
            datasets[dset_type] = MyDataset(test_df, transform = transform)
            loaders[dset_type] = torch.utils.data.DataLoader(datasets[dset_type], batch_size=batchSize, shuffle=False)
        print('Finished loading %s dataset: %s samples' % (dset_type, len(datasets[dset_type])))
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

    batchSize=80
    imgSize=int(299)
    method = "mean"

    model = models.mobilenet_v2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = True
    model.classifier = nn.Sequential(nn.Dropout(p=0.7), nn.Linear(1280, 1))
    model = model.to('cuda')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    params_to_update = model.parameters()
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = optim.AdamW(params_to_update, lr=0.001)

    model, state_dicts, best_loss, best_cindex, best_epoch = train_model(model, loaders, optimizer, scheduler=None, num_epochs=30)
    torch.save(model.state_dict(), '/path/to/save/checkpoints')

    output_test, event_test, time_test, loss_test, pts_test = test_model(model, loaders['test'], dataset_sizes['test'])

    df = pd.DataFrame(columns=['id', 'output', 'event', 'time'])
    df.id = pts_test
    df.output = output_test
    df.event = event_test
    df.time = time_test
    unique=np.unique(df.id.values).tolist()

    pt_output=[]
    pt_event=[]
    pt_time=[]
    for i in range(len(unique)):
        ave_output=aggoutputs(df[df.id==unique[i]].output.values, method)
        ev=df[df.id==unique[i]].event.values.tolist()[0]
        ti=df[df.id==unique[i]].time.values.tolist()[0]
        pt_output.append(ave_output)
        pt_event.append(ev)
        pt_time.append(ti)
    dd = pd.DataFrame(columns=['id', 'output', 'event', 'time'])
    dd.id = unique
    dd.output=pt_output
    dd.event=pt_event
    dd.time=pt_time
    pt_output = torch.tensor(pt_output).cuda()
    pt_event = torch.tensor(pt_event).cuda()
    pt_time = torch.tensor(pt_time).cuda()
    cindex, concordant, discordant, tied_risk, tied_time, _, _ = concordance_index_censored(pt_event, pt_time, pt_output, tied_tol=1e-8)
    print("Harrell's C-index = " + str(cindex))
    print("Concordant = " + str(concordant))
    print("Discordant = " + str(discordant))
    print("Tied risk = " + str(tied_risk))
    print("Tied time = " + str(tied_time))

    dev_event = np.concatenate((datasets['train'].df.event.values, datasets['val'].df.event.values))
    dev_time = np.concatenate((datasets['train'].df.time.values, datasets['val'].df.time.values))
    _dev_event = [bool(i) for i in dev_event]
    dev_data = np.array([(i, j) for i, j in zip(_dev_event, dev_time)],dtype=[('event', '?'), ('time', '<f8')])
    _pt_event = [bool(i) for i in pt_event.cpu()]
    pt_data = np.array([(i, j) for i, j in zip(_pt_event, pt_time.cpu())],dtype=[('event', '?'), ('time', '<f8')])
    cindex2, concordant2, discordant2, tied_risk2, tied_time2 = concordance_index_ipcw(dev_data, pt_data, pt_output.cpu(), tau=None, tied_tol=1e-08)
    print("Uno's C-index = " + str(cindex2))
    print("Concordant = " + str(concordant2))
    print("Discordant = " + str(discordant2))
    print("Tied risk = " + str(tied_risk2))
    print("Tied time = " + str(tied_time2))