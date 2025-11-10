import os
import csv
import json
import pickle
import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from random import shuffle
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import confusion_matrix

def separability(X, y):
    X_proj = X
    y = np.array(y)
    sep_ = {}
    class_sizes = {}
    var_inter = {}
    var_intra = {}
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        class_sizes[c] = len(idx_c)
        inter_var = []
        for c2 in np.unique(y):
            if c != c2:
               idx_c2 = np.where(y == c2)[0]
               X_ = np.r_[X_proj[idx_c], X_proj[idx_c2]]
               inter_var.append(np.mean(np.var(X_, axis=0)))
        mean_intra = np.mean(np.var(X_proj[idx_c], axis=0))
        # print('  *', c, len(idx_c), '|', mean_intra, np.mean(inter_var), '|', np.var(X_proj[idx_c], axis=0), inter_var)
        var_inter[c] = np.mean(inter_var)
        var_intra[c] = mean_intra
        if(mean_intra != 0):
            sep_[c] = np.mean(inter_var) / mean_intra
        else:
            print('mean_intra = 0')
            sep_[c] = np.mean(inter_var)
    # # experimental ! 
    # for c in class_sizes.keys():
    #     sep_[c] *= class_sizes[c] / np.sum([class_sizes[k] for k in class_sizes.keys()])
    return sep_, var_inter, var_intra

def consistency(X, y):
    y = np.array(y)
    consist = {}
    var_intra = {}
    var_inter = {}

    class_indexes = {}
    for c in np.unique(y):
        class_indexes[c] = np.where(y == c)[0]

    sep_ = {}
    for c in np.unique(y):
        sep_[c] = []

    for l in range(0, len(class_indexes[c]), 2):
        for c in np.unique(y):
            # print(len(class_indexes[c]), len(class_indexes[c][l:]))
            if len(class_indexes[c][l:]) >= 2:
                idx_c = [class_indexes[c][l], class_indexes[c][l+1]]
            # else:
            #     idx_c = [class_indexes[c][l]]
                inter_var = []
                for c2 in np.unique(y):
                    if c != c2:
                        # print(c, c2, l, len(class_indexes[c2]), len(class_indexes[c2][l:]))
                        if len(class_indexes[c2][l:]) >= 2:
                            idx_c2 = [class_indexes[c2][l], class_indexes[c2][l+1]]
                            X_ = np.r_[X[idx_c], X[idx_c2]]
                            inter_var.append(np.mean(np.var(X_, axis=0)))
                        else:
                            if len(class_indexes[c2][l:]) > 0:
                                idx_c2 =  [class_indexes[c2][l]]
                                X_ = np.r_[X[idx_c], X[idx_c2]]
                                inter_var.append(np.mean(np.var(X_, axis=0)))
                            # else:
                            #     idx_c2 =  []
                            #     inter_var.append(
                    
            mean_intra = np.mean(np.var(X[idx_c], axis=0))
            # print('  *', c, len(idx_c), '|', mean_intra, np.mean(inter_var), '|', np.var(X_proj[idx_c], axis=0), inter_var)
            var_inter[c] = np.mean(inter_var)
            var_intra[c] = mean_intra
            if(mean_intra != 0):
                sep_[c].append(np.mean(inter_var) / mean_intra)
            else:
                print('mean_intra = 0')
                sep_[c].append(np.mean(inter_var))
    
    for c in np.unique(y):
        # print(c, sep_[c])
        sep_[c] = np.mean(sep_[c])
        # print(c, sep_[c])
        # intra_sum = 0
        # intra_tot = 0
        # if len(idx_c) >= 2:
        #     for i in range(0, len(idx_c), 2):
        #         intra_sum += np.sqrt(np.sum(np.power(X[idx_c[i]] - X[idx_c[i+1]], 2))) / 8
        #         intra_tot += 1
        # else:
        #     intra_tot = 1
        # mean_vec_c = np.mean(X[idx_c], axis=0)
        # # var_intra[c] = np.mean([np.sqrt(np.sum(np.power(X[i] - mean_vec_c, 2))) for i in idx_c], axis=0)
        # var_intra[c] = intra_sum / intra_tot
        # # print(len(idx_c), var_intra[c])
        # inter = []
        # for c2 in np.unique(y):
        #     if c != c2:
        #         idx_c2 = np.where(y == c2)[0]
        #         mean_vec_c2 = np.mean(X[idx_c2], axis=0)
        #         #    inter.append(np.mean(
        #         # #     [np.sqrt(np.sum(np.power(X[i] - mean_vec_c2, 2))) for i in idx_c2], axis=0))
        #         inter.append(np.sqrt(np.sum(np.power(mean_vec_c - mean_vec_c2, 2))))
        # var_inter[c] = np.mean(inter)
        # if(var_intra[c] != 0):
        #     consist[c] = var_inter[c] / var_intra[c]
        # else:
        #     print('mean_intra = 0')
        #     consist[c] = var_inter[c]
        
    return sep_


def confusions(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes).T
    entropies = {}
    for ci, c in enumerate(classes):
        if np.sum(cm[ci]) > 0:
            entropies[c] = stats.entropy(cm[ci] / np.sum(cm[ci]))
        else:
            entropies[c] = 0.0
        # print(c, cm[ci], entropies[c])
        # print(cm)
    return entropies #stats.entropy(np.diagonal(cm) / np.sum(np.diagonal(cm)))

def confusions2(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes).T
    c_ = [int(c == 6) for c in np.diagonal(cm)]
    p = np.sum(c_) / 8
    print(cm, c_, p)
    # entropies = {}
    # for ci, c in enumerate(classes):
    #     if np.sum(cm[ci]) > 0:
    #         entropies[c] = stats.entropy(cm[ci] / np.sum(cm[ci]))
    #     else:
    #         entropies[c] = 0.0
    #     # print(c, cm[ci], entropies[c])
    #     # print(cm)
    return p #stats.entropy(np.diagonal(cm) / np.sum(np.diagonal(cm)))


def remove_outliers(data, factors, measure):
    new_data = {}
    for k in data.keys():
        new_data[k] = []
    if len(factors) == 2:
        for c in np.unique(data[factors[0]]):
            for c2 in np.unique(data[factors[1]]):
                idx1 = np.where(np.array(data[factors[0]]) == c)[0]
                idx2 = np.where(np.array(data[factors[1]]) == c2)[0]
                idx = list(set(idx1) & set(idx2))
                m = np.mean(np.array(data[measure])[idx])
                s = np.std(np.array(data[measure])[idx])
                nidx = []
                for i in idx:
                    if np.array(data[measure])[i] <= m + 3*s and np.array(data[measure])[i] >= m - 3*s:
                        nidx.append(i)
                for k in data.keys():
                    new_data[k].extend(np.array(data[k])[nidx])
                print(c, c2, len(nidx), len(idx))
    elif len(factors) == 1:
        for c in np.unique(data[factors[0]]):
            idx = np.where(np.array(data[factors[0]]) == c)[0]
            m = np.mean(np.array(data[measure])[idx])
            s = np.std(np.array(data[measure])[idx])
            nidx = []
            for i in idx:
                if np.array(data[measure])[i] <= m + 3*s and np.array(data[measure])[i] >= m - 3*s:
                    nidx.append(i)
            for k in data.keys():
                new_data[k].extend(np.array(data[k])[nidx])
            if len(nidx)==len(idx):
                print(c, '- no outlier removed -', len(nidx), len(idx))
            else:
                print(c, '- {} outliers removed -'.format(len(idx)-len(nidx)), len(nidx), len(idx))
    return new_data

def load_data(data_path):
    '''
    Load data from a folder organised by COND and PID with .db files and models/ folder
    Take the folder name to browse as arguments
    Return the data and the models 
    '''
    res = []
    models = {}
    data = {}
    # Iterate directory
    for root, dirs, files in os.walk(data_path):
        for file in files:
            # handle db files (movement data basically)
            if file.endswith(".db"):
                # participant id 
                pid = int(root.split('/')[2][1:])
                # condition
                cond = root.split('/')[1]
                # data structure 
                if(pid not in data.keys()):
                    data[pid] = {}
                    data[pid]['cond'] = cond
                    data[pid]['training'] = {}
                    data[pid]['training']['x'] = []
                    data[pid]['training']['y'] = []
                    data[pid]['posttest'] = {}
                    data[pid]['posttest']['x'] = []
                    data[pid]['posttest']['y'] = []
                    data[pid]['negative'] = {}
                    data[pid]['negative']['x'] = []
                    data[pid]['negative']['y'] = []
                    data[pid]['positive'] = {}
                    data[pid]['positive']['x'] = []
                    data[pid]['positive']['y'] = []
                with open(os.path.join(root, file)) as f:  
                    content = f.read().splitlines()
                    if('training' in file):
                        times = {}
                        for j in range(len(content)):
                            content_json = json.loads(content[j])
                            times[content_json['createdAt']['$$date']] = j
                        for k in sorted(times.keys()):
                            content_json = json.loads(content[times[k]])
                            y = content_json['y']
                            x = content_json['x']
                            data[pid]['training']['x'].append(x)
                            data[pid]['training']['y'].append(y)
                    elif('posttest' in file):
                        for j in range(len(content)):
                            content_json = json.loads(content[j])
                            y = content_json['y']
                            x = content_json['x']
                            data[pid]['posttest']['x'].append(x)
                            data[pid]['posttest']['y'].append(y)
                    elif('pos-neg' in file):
                        for j in range(len(content)):
                            content_json = json.loads(content[j])
                            y = content_json['y']
                            x = content_json['x']
                            # data[pid]['posttest']['x'].append(x)
                            # data[pid]['posttest']['y'].append(y)
                            if 'Negative' in y:
                                ny = ' '.join(y.split(' ')[:-1])
                                data[pid]['negative']['x'].append(x)
                                data[pid]['negative']['y'].append(ny)
                            elif 'Positive' in y:
                                ny = ' '.join(y.split(' ')[:-1])
                                data[pid]['positive']['x'].append(x)
                                data[pid]['positive']['y'].append(ny)

            # handle pickle files (models)
            if file.endswith(".pickle") and file.split('_')[0] == 'model':
                # participant id 
                pid = int(root.split('/')[2][1:])
                phase_id = int(file.split('_')[1][-1])
                # print(pid, phase_id)
                if phase_id == 2:
                    if pid not in models.keys():
                        models[pid] = {}
                        models[pid]['training'] = {}
                    trial_id = int(file.split('_')[2])
                    # print(file, pid, phase_id, trial_id)
                    models[pid]['training'][trial_id] = pickle.load(open(os.path.join(root, file), 'rb'))
                elif phase_id == 1:
                    if pid not in models.keys():
                        models[pid] = {}
                        # models[pid]['init'] = {}
                    # if 'init' not in models[pid].keys():
                    #     models[pid]['init'] = {}
                    models[pid]['init'] = pickle.load(open(os.path.join(root, file), 'rb'))
    # check outlier baased on training accuracy
    data_sns = {'pid': [], 'condition': [], 'accuracy': []}
    outliers = []
    for pid in data.keys():
        classifier = LinearDiscriminantAnalysis()
        X = np.array(data[pid]['training']['x'])
        y = np.array(data[pid]['training']['y'])
        classifier.fit(X, y)
        data_sns['pid'].append(pid)
        data_sns['condition'].append(data[pid]['cond'])
        data_sns['accuracy'].append(
        classifier.score(
            np.array(data[pid]['posttest']['x']),
            np.array(data[pid]['posttest']['y'])))
    stds = {}
    meas = {}
    for c in np.unique(data_sns['condition']):
        idx = np.where(np.array(data_sns['condition']) == c)[0]
        acs = [data_sns['accuracy'][i] for i in idx]
        stds[c] = np.std(acs)
        meas[c] = np.mean(acs)
        for i in idx:
            if data_sns['accuracy'][i] <= meas[c] - 3*stds[c]:
                print('Outlier detected', c, data_sns['pid'][i], data_sns['accuracy'][i], meas[c], stds[c])
                outliers.append(data_sns['pid'][i])
    data2 = {}
    for pid in data.keys():
        if pid not in outliers:
            data2[pid] = data[pid]
    return data2, models

def perclass_accuracy(y_true, y_pred, classes, metric='acc'):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    true_pos = {}
    true_neg = {}
    pos = {}
    neg = {}
    false_pos = {}
    false_pos = {}
    true_pos_rate = {}
    for r in range(len(cm)):
        total = 0
        tn = 0
        for c in range(len(cm)):
            if r == c:
                diag = cm[r,c]
                true_pos[r] = diag
            total += cm[r,c]

        pos[r]  = total
        
        # print(r, diag, total)
        # if total != 0:
        true_pos_rate[r] = diag / total
        
        # else:
        #     true_pos_rate[r] = 1.0

    return true_pos_rate, true_pos, pos

def sanity_check(data):
    for pid in data.keys():
        for phase in data[pid].keys():
            if phase != 'cond':
                for i in range(len(data[pid][phase]['x'])):
                    if len(data[pid][phase]['x'][i]) != 8:
                        print('pb', pid, phase, i, len(data[pid][phase]['x'][i]))
                    else:
                        if np.var(data[pid][phase]['x'][i]) < 0.1:
                            print('pb', pid, phase, i, np.var(data[pid][phase]['x'][i]), data[pid][phase]['x'][i])
                    # print(pid, phase, data[pid][phase]['x'][i])

def check_init_acc(data, 
                 models,perclass = True):
  
    acc = []
    if perclass:
        data_sns = {'pid': [], 'condition': [], 'class': [], 'accuracy': []}
    else:
        data_sns = {'pid': [], 'condition': [], 'accuracy': []}
    for pid in data.keys():
        print(data[pid]['cond'])
        X = np.array(data[pid]['training']['x'])
        y = np.array(data[pid]['training']['y'])
    
        acc_ = []
            
        classifier = models[pid]['init']
        # classifier = LinearDiscriminantAnalysis()
        X = np.array(data[pid]['training']['x'])[0:16,:]
        y = np.array(data[pid]['training']['y'])[0:16]
        # classifier.fit(X, y)
        
        if not perclass:
            
            data_sns['pid'].append(pid)
            data_sns['condition'].append(data[pid]['cond'])
            data_sns['accuracy'].append(
                classifier.score(
                    np.array(data[pid]['posttest']['x']),
                    np.array(data[pid]['posttest']['y'])))
        else:
            
            cm = confusion_matrix(
                np.array(data[pid]['posttest']['y']), 
                classifier.predict(np.array(data[pid]['posttest']['x'])), 
                labels=classifier.classes_)
            
            acc_2 = []
            for r in range(len(cm)):
                total = 0
                for c in range(len(cm)):
                    if r == c:
                        diag = cm[r,c]
                    total += cm[r,c]
                data_sns['pid'].append(pid)
                data_sns['condition'].append(data[pid]['cond'])
                data_sns['class'].append(classifier.classes_[r])
                if total != 0:
                    data_sns['accuracy'].append(diag/total)
                else:
                    data_sns['accuracy'].append(1.0)
                acc_2.append(diag/total)
            acc_.append(np.mean(acc_2))        
    acc.append(acc_)          
        
    # acc = np.array(acc)
    
    # plt.plot(np.mean(acc, axis=0), '-')
    # plt.show()

    # new_data = data_sns
 
    print(data_sns['condition'])
    df = pd.DataFrame(data_sns)
    mean_rc = df[df['condition']=='NUC']['accuracy'].mean()
    mean_tlc = df[df['condition']=='UC']['accuracy'].mean()
    mean_llc = df[df['condition']=='NUCS']['accuracy'].mean()
    print(mean_rc,mean_tlc,mean_llc)
    sns.barplot(data=df, x='condition', y='accuracy', hue='condition')
    plt.show()

def check_uc_acc(data, 
                 models):

    data_sns = {'pid': [], 'condition': [], 'class': [], 'trial': [], 'accuracy': []}
    perclass = True 
    acc = []
    for pid in data.keys():

        

        if data[pid]['cond'] == 'UC':

            X = np.array(data[pid]['training']['x'])
            y = np.array(data[pid]['training']['y'])
        
            acc_ = []

            for trial in range(17,len(X)):
                
                # classifier = models[pid]['training'][trial]
                classifier = LinearDiscriminantAnalysis()
                X = np.array(data[pid]['training']['x'])[:trial,:]
                y = np.array(data[pid]['training']['y'])[:trial]
                classifier.fit(X, y)
                
                if not perclass:
                    data_sns['pid'].append(pid)
                    data_sns['condition'].append(data[pid]['cond'])
                    data_sns['trial'].append(trial-16)
                    data_sns['accuracy'].append(
                        classifier.score(
                            np.array(data[pid]['posttest']['x']),
                            np.array(data[pid]['posttest']['y'])))
                else:
                    cm = confusion_matrix(
                        np.array(data[pid]['posttest']['y']), 
                        classifier.predict(np.array(data[pid]['posttest']['x'])), 
                        labels=classifier.classes_)
                    print(pid, trial, cm)
                    acc_2 = []
                    for r in range(len(cm)):
                        total = 0
                        for c in range(len(cm)):
                            if r == c:
                                diag = cm[r,c]
                            total += cm[r,c]
                        data_sns['pid'].append(pid)
                        data_sns['condition'].append(data[pid]['cond'])
                        data_sns['trial'].append(trial-16)
                        data_sns['class'].append(classifier.classes_[r])
                        if total != 0:
                            data_sns['accuracy'].append(diag/total)
                        else:
                            data_sns['accuracy'].append(1.0)
                        acc_2.append(diag/total)
                    acc_.append(np.mean(acc_2))        
            acc.append(acc_)          
            # plt.plot(acc_, '-')
            # plt.show()
    acc = np.array(acc)
    print(acc.shape)
    plt.plot(np.mean(acc, axis=0), '-')
    plt.show()

    new_data = data_sns
    
    df = pd.DataFrame(data_sns)
    
    sns.lineplot(data=df, x='trial', y='accuracy', hue='condition')
    plt.show()

def compute_clf_accuracy(data, 
                         models,
                         train_phase,
                         test_phase,
                         model_t,
                         perclass,
                         offset, 
                         type_of_plot):
    # model specified: load model at the given training trial t
    if model_t != None:

        # depending if we consider per-class accuracy or mean accuracy over classes
        if not perclass:
            data_sns = {'pid': [], 'condition': [], 'accuracy': []}
        else:
            data_sns = {'pid': [], 'condition': [], 'class': [], 'accuracy': []}
        
        # main loop on participant id
        for pid in data.keys():
            
            # if no models provided, train a new one (not recommended!!)
            if models == None:
                classifier = LinearDiscriminantAnalysis()
                X = np.array(data[pid][train_phase]['x'])[:model_t,:]
                y = np.array(data[pid][train_phase]['y'])[:model_t]
                classifier.fit(X, y)
            else:
                classifier = models[pid]['training'][model_t]
            
            # condition over the test set to use: posttest or some part of the training set
            if test_phase == 'posttest':
                if not perclass:
                    data_sns['pid'].append(pid)
                    data_sns['condition'].append(data[pid]['cond'])
                    data_sns['accuracy'].append(
                        classifier.score(
                            np.array(data[pid][test_phase]['x']),
                            np.array(data[pid][test_phase]['y'])))
                else:
                    perclass_acc,_,_ = perclass_accuracy(
                        np.array(data[pid][test_phase]['y']),
                        classifier.predict(np.array(data[pid][test_phase]['x'])), 
                        classifier.classes_
                    )
                    print(perclass_acc)
                    for k in perclass_acc.keys():
                        data_sns['pid'].append(pid)
                        data_sns['condition'].append(data[pid]['cond'])
                        data_sns['class'].append(k)
                        data_sns['accuracy'].append(perclass_acc[k])

            # if the test set used it the training set, either we take all the data after the model id (not seen data) 
            # or if the model id is the last one (at t = 120), we take the last 5 instances per class of the training phase...
            elif test_phase == 'training':
                if model_t < 120:
                    x_eot = np.array(data[pid]['training']['x'])[model_t:,:]
                    y_eot = np.array(data[pid]['training']['y'])[model_t:]
                else:
                    x_eot, y_eot = [], []
                    for c in np.unique(data[pid]['training']['y']):
                        idx = np.where(np.array(data[pid]['training']['y']) == c)[0]
                        x_eot.extend(np.array(data[pid]['training']['x'])[idx[len(idx) - 3:],:])
                        y_eot.extend(np.array(data[pid]['training']['y'])[idx[len(idx) - 3:]])
                    x_eot = np.array(x_eot)
                    y_eot = np.array(y_eot)
                    # x_eot = np.array(data[pid][train_phase]['x'])
                    # y_eot = np.array(data[pid][train_phase]['y'])
                
                # same than above, accuracy is eithr computed over all classes or per class
                if not perclass:
                    data_sns['pid'].append(pid)
                    data_sns['condition'].append(data[pid]['cond'])
                    data_sns['accuracy'].append(classifier.score(x_eot, y_eot))
                else:
                    perclass_acc = perclass_accuracy(
                        y_eot,
                        classifier.predict(x_eot), 
                        classifier.classes_
                    )
                    for k in perclass_acc.keys():
                        data_sns['pid'].append(pid)
                        data_sns['condition'].append(data[pid]['cond'])
                        data_sns['class'].append(k)
                        data_sns['accuracy'].append(perclass_acc[k])

        new_data = remove_outliers(
            data_sns, 
            factors=['condition'], 
            measure='accuracy')

        with open('accuracies_{}.csv'.format(test_phase), 'w') as f:  # You will need 'wb' mode in Python 2.x
            keys = new_data.keys()
            w = csv.DictWriter(f, keys)
            w.writeheader()
            for l in range(len(new_data['pid'])):
                row_to_write = {}
                for k in keys:
                    row_to_write[k] = new_data[k][l]
                w.writerow(row_to_write)


        df = pd.DataFrame(new_data)
        fval, pval = stats.f_oneway(
            df[df['condition'] == 'NUC']['accuracy'],
            df[df['condition'] == 'UC']['accuracy'],
            df[df['condition'] == 'NUCS']['accuracy'])
        print('** one-way ANOVA')
        model = ols('accuracy ~ C(condition)', data=df).fit()
        print(sm.stats.anova_lm(model, typ=1)) 
        print('** pairwise t-tests')
        for ci, c in enumerate(np.unique(new_data['condition'])):
            for c2i, c2 in enumerate(np.unique(new_data['condition'])):
                if c2i > ci:
                    print('{} - {}\t'.format(c,c2), 
                          stats.ttest_ind(df[df['condition'] == c]['accuracy'], df[df['condition'] == c2]['accuracy']))
        
        for k in np.unique(new_data['condition']):
            print(k, np.mean(df[df['condition'] == k]['accuracy']))

        if type_of_plot == 'violinplot':
            sns.violinplot(data=df, x='condition', y='accuracy', inner="points")
        elif type_of_plot == 'barplot':
            sns.barplot(data=df, x='condition', y='accuracy')
        plt.title("retrained models - anova's pvalue={:.3f}".format(pval))
        plt.savefig('model_accuracy_testset={}_perclass={}_plot={}.png'.format(test_phase, perclass, type_of_plot))
        plt.show()

    else:
        if not perclass:
            data_sns = {'pid': [], 'condition': [], 'trial': [], 'accuracy': []}
        else:
            data_sns = {'pid': [], 'condition': [], 'class': [], 'trial': [], 'accuracy': []}
        for pid in data.keys():
            X = np.array(data[pid][train_phase]['x'])
            y = np.array(data[pid][train_phase]['y'])
            for trial in range(17,len(X)):
                if models == None:
                    classifier = LinearDiscriminantAnalysis()
                    classifier.fit(X[:trial,:], y[:trial])
                else:
                    classifier = models[pid]['training'][trial]
                if not perclass:
                    data_sns['pid'].append(pid)
                    data_sns['condition'].append(data[pid]['cond'])
                    data_sns['trial'].append(trial-16)
                    data_sns['accuracy'].append(
                        classifier.score(
                            np.array(data[pid][test_phase]['x']),
                            np.array(data[pid][test_phase]['y'])))
                else:
                    cm = confusion_matrix(
                        np.array(data[pid][test_phase]['y']), 
                        classifier.predict(np.array(data[pid][test_phase]['x'])), 
                        labels=classifier.classes_)
                    for r in range(len(cm)):
                        total = 0
                        for c in range(len(cm)):
                            if r == c:
                                diag = cm[r,c]
                            total += cm[r,c]
                        data_sns['pid'].append(pid)
                        data_sns['condition'].append(data[pid]['cond'])
                        data_sns['trial'].append(trial-16)
                        data_sns['class'].append(classifier.classes_[r])
                        if total != 0:
                            data_sns['accuracy'].append(diag/total)
                        else:
                            data_sns['accuracy'].append(1.0)

        filename ='accuracy_along_training' 
        for k in np.unique(data_sns['condition']):
            idx = np.where(np.array(data_sns['condition']) == k)[0]
            m = np.mean(np.array(data_sns['accuracy'])[idx])
            print(k, m)

        if offset:
            filename += '_offset'
            for k in np.unique(data_sns['condition']):
                idxc = np.where(np.array(data_sns['condition']) == k)[0]
                idxt = np.where(np.array(data_sns['trial']) == 1)[0]
                idx = list(set(idxc) & set(idxt))
                m = np.mean(np.array(data_sns['accuracy'])[idx])
                print(c, m)
                for i in idxc:
                    data_sns['accuracy'][i] = data_sns['accuracy'][i] - m
        new_data = data_sns
        # new_data = remove_outliers(
        #     data_sns, 
        #     factors = ['condition', 'trial'], 
        #     measure='accuracy')

        # new_data = data_sns
        
        with open(filename+'.csv', 'w') as f: 
            keys = new_data.keys()
            w = csv.DictWriter(f, keys)
            w.writeheader()
            for l in range(len(new_data['pid'])):
                row_to_write = {}
                for k in keys:
                    row_to_write[k] = new_data[k][l]
                w.writerow(row_to_write)

        df = pd.DataFrame(data_sns)
        # for t in sorted(np.unique(data_sns['trial'])):
        #     fval, pval = stats.f_oneway(
        #         df[(df['trial'] == t) & (df['condition'] == 'NUC')]['accuracy'],
        #         df[(df['trial'] == t) & (df['condition'] == 'NUCS')]['accuracy'],
        #         df[(df['trial'] == t) & (df['condition'] == 'UC')]['accuracy'])
        #     print(t, '\t', pval)
        sns.lineplot(data=df, x='trial', y='accuracy', hue='condition')
        plt.title('retrained models')
        plt.savefig('compute_clf_accuracy_along_training_perclass={}.png'.format(perclass))
        plt.show()

def compute_confusion(data, 
                      models,
                      train_phase,
                      test_phase,
                      perclass,
                      model_t):

    # model specified: load model at the given training trial t
    if model_t != None:

        # depending if we consider per-class confusion or mean confusion over classes
        if not perclass:
            data_sns = {'pid': [], 'condition': [], 'confusion': []}
        else:
            data_sns = {'pid': [], 'condition': [], 'class': [], 'confusion': []}
        
        # main loop on participant id
        for pid in data.keys():
            
            classifier = models[pid]['training'][model_t]
            
            # condition over the test set to use: posttest or some part of the training set
            if test_phase == 'posttest':
                entropies = confusions(
                        classifier.predict(np.array(data[pid][test_phase]['x'])),
                        np.array(data[pid][test_phase]['y']),
                        classes=classifier.classes_)
                if perclass:
                    for k in entropies.keys():
                        data_sns['pid'].append(pid)
                        data_sns['condition'].append(data[pid]['cond'])
                        data_sns['class'].append(k)
                        data_sns['confusion'].append(entropies[k])
                else:
                    # e_ = np.mean([entropies[k] for k in entropies.keys()])
                    e_ = confusions2(
                        classifier.predict(np.array(data[pid][test_phase]['x'])),
                        np.array(data[pid][test_phase]['y']),
                        classes=classifier.classes_)
                    # for k in entropies.keys():
                    data_sns['pid'].append(pid)
                    data_sns['condition'].append(data[pid]['cond'])
                    data_sns['confusion'].append(e_)
                # else:
                #     perclass_acc = perclass_accuracy(
                #         np.array(data[pid][test_phase]['y']),
                #         classifier.predict(np.array(data[pid][test_phase]['x'])), 
                #         classifier.classes_
                #     )
                #     for k in perclass_acc.keys():
                #         data_sns['pid'].append(pid)
                #         data_sns['condition'].append(data[pid]['cond'])
                #         data_sns['class'].append(k)
                #         data_sns['accuracy'].append(perclass_acc[k])

            # # if the test set used it the training set, either we take all the data after the model id (not seen data) 
            # # or if the model id is the last one (at t = 120), we take the last 5 instances per class of the training phase...
            # elif test_phase == 'training':
            #     if model_t < 120:
            #         x_eot = np.array(data[pid]['training']['x'])[model_t:,:]
            #         y_eot = np.array(data[pid]['training']['y'])[model_t:]
            #     else:
            #         x_eot, y_eot = [], []
            #         for c in np.unique(data[pid]['training']['y']):
            #             idx = np.where(np.array(data[pid]['training']['y']) == c)[0]
            #             x_eot.extend(np.array(data[pid]['training']['x'])[idx[len(idx) - 5:],:])
            #             y_eot.extend(np.array(data[pid]['training']['y'])[idx[len(idx) - 5:]])
            #         x_eot = np.array(x_eot)
            #         y_eot = np.array(y_eot)
                
            #     # same than above, accuracy is eithr computed over all classes or per class
            #     if not perclass:
            #         data_sns['pid'].append(pid)
            #         data_sns['condition'].append(data[pid]['cond'])
            #         data_sns['accuracy'].append(classifier.score(x_eot, y_eot))
            #     else:
            #         perclass_acc = perclass_accuracy(
            #             y_eot,
            #             classifier.predict(x_eot), 
            #             classifier.classes_
            #         )
            #         for k in perclass_acc.keys():
            #             data_sns['pid'].append(pid)
            #             data_sns['condition'].append(data[pid]['cond'])
            #             data_sns['class'].append(k)
            #             data_sns['accuracy'].append(perclass_acc[k])

        # new_data = remove_outliers(
        #     data_sns, 
        #     factors=['condition'], 
        #     measure='confusion')
        new_data = data_sns
        # print(new_data)

        with open('confusions_{}.csv'.format(test_phase), 'w') as f:  
            keys = new_data.keys()
            w = csv.DictWriter(f, keys)
            w.writeheader()
            for l in range(len(new_data['pid'])):
                row_to_write = {}
                for k in keys:
                    row_to_write[k] = new_data[k][l]
                w.writerow(row_to_write)

        df = pd.DataFrame(new_data)
        fval, pval = stats.f_oneway(
            df[df['condition'] == 'NUC']['confusion'],
            df[df['condition'] == 'UC']['confusion'],
            df[df['condition'] == 'NUCS']['confusion'])
        print('** one-way ANOVA')
        model = ols('confusion ~ C(condition)', data=df).fit()
        print(sm.stats.anova_lm(model, typ=1)) 
        print('** pairwise t-tests')
        for ci, c in enumerate(np.unique(new_data['condition'])):
            for c2i, c2 in enumerate(np.unique(new_data['condition'])):
                if c2i > ci:
                    print('{} - {}\t'.format(c,c2), 
                          stats.ttest_ind(df[df['condition'] == c]['confusion'], df[df['condition'] == c2]['confusion']))
        
        for k in np.unique(new_data['condition']):
            print(k, np.mean(df[df['condition'] == k]['confusion']))

        # if type_of_plot == 'violinplot':
        #     sns.violinplot(data=df, x='condition', y='confusion', inner="points")
        # elif type_of_plot == 'barplot':
        sns.violinplot(data=df, x='condition', y='confusion', inner="points")
        plt.title("retrained models - anova's pvalue={:.3f}".format(pval))
        # plt.savefig('model_confusion_testset={}_plot={}.png'.format(test_phase,  type_of_plot))
        plt.show()

    # else:
    #     if not perclass:
    #         data_sns = {'pid': [], 'condition': [], 'trial': [], 'accuracy': []}
    #     else:
    #         data_sns = {'pid': [], 'condition': [], 'class': [], 'trial': [], 'accuracy': []}
    #     for pid in data.keys():
    #         X = np.array(data[pid][train_phase]['x'])
    #         y = np.array(data[pid][train_phase]['y'])
    #         for trial in range(17,len(X)):
    #             if models == None:
    #                 classifier = LinearDiscriminantAnalysis()
    #                 classifier.fit(X[:trial,:], y[:trial])
    #             else:
    #                 classifier = models[pid]['training'][trial]
    #             if not perclass:
    #                 data_sns['pid'].append(pid)
    #                 data_sns['condition'].append(data[pid]['cond'])
    #                 data_sns['trial'].append(trial-16)
    #                 data_sns['accuracy'].append(
    #                     classifier.score(
    #                         np.array(data[pid][test_phase]['x']),
    #                         np.array(data[pid][test_phase]['y'])))
    #             else:
    #                 cm = confusion_matrix(
    #                     np.array(data[pid][test_phase]['y']), 
    #                     classifier.predict(np.array(data[pid][test_phase]['x'])), 
    #                     labels=classifier.classes_)
    #                 for r in range(len(cm)):
    #                     total = 0
    #                     for c in range(len(cm)):
    #                         if r == c:
    #                             diag = cm[r,c]
    #                         total += cm[r,c]
    #                     data_sns['pid'].append(pid)
    #                     data_sns['condition'].append(data[pid]['cond'])
    #                     data_sns['trial'].append(trial-16)
    #                     data_sns['class'].append(classifier.classes_[r])
    #                     if total != 0:
    #                         data_sns['accuracy'].append(diag/total)
    #                     else:
    #                         data_sns['accuracy'].append(1.0)

    #     df = pd.DataFrame(data_sns)
    #     for t in sorted(np.unique(data_sns['trial'])):
    #         fval, pval = stats.f_oneway(
    #             df[(df['trial'] == t) & (df['condition'] == 'NUC')]['accuracy'],
    #             df[(df['trial'] == t) & (df['condition'] == 'NUCS')]['accuracy'],
    #             df[(df['trial'] == t) & (df['condition'] == 'UC')]['accuracy'])
    #         print(t, '\t', pval)
    #     sns.lineplot(data=df, x='trial', y='accuracy', hue='condition')
    #     plt.title('retrained models')
    #     plt.savefig('compute_clf_accuracy_along_training_perclass={}.png'.format(perclass))
    #     plt.show()

def compute_separability(data, 
                         models, 
                         type_of_sep,
                         data_phase, 
                         model_t, 
                         perclass,
                         log_scale):
    # depending on whether we consider per-class accuracy or mean accuracy over classes
    if model_t != None:
        data_sns = {
            'pid': [], 
            'condition': [], 
            'separability': [], 
            'var_inter': [], 
            'var_intra': []}
        if perclass:
            data_sns['class'] = []
        
        for pid in data.keys():
            X = np.array(data[pid][data_phase]['x'])
            y = np.array(data[pid][data_phase]['y'])
            if data_phase == 'training':
                X = X[:model_t,:]
                y = y[:model_t]
            classifier = models[pid]['training'][model_t]
            if type_of_sep == 'lda':
                sep_perclass, var_inter, var_intra = separability(classifier.transform(X), y)
            elif type_of_sep == 'raw':
                sep_perclass, var_inter, var_intra = separability(X, y)

            if not perclass:
                sep = np.mean([sep_perclass[c] for c in sep_perclass.keys()])
                vinter = np.mean([var_inter[c] for c in var_inter.keys()])
                vintra = np.mean([var_intra[c] for c in var_intra.keys()])
                data_sns['pid'].append(pid)
                data_sns['condition'].append(data[pid]['cond'])
                if not log_scale:
                    data_sns['separability'].append(sep)
                    data_sns['var_inter'].append(vinter)
                    data_sns['var_intra'].append(vintra)
                else:
                    data_sns['separability'].append(np.log(sep))
                    data_sns['var_inter'].append(np.log(vinter))
                    data_sns['var_intra'].append(np.log(vintra))
            else:
                for c in sep_perclass.keys():
                    data_sns['pid'].append(pid)
                    data_sns['class'].append(list(classifier.classes_).index(c))
                    data_sns['condition'].append(data[pid]['cond'])
                    if not log_scale:
                        data_sns['separability'].append(sep_perclass[c])
                        data_sns['var_inter'].append(var_inter[c])
                        data_sns['var_intra'].append(var_intra[c])
                    else:
                        data_sns['separability'].append(np.log(sep_perclass[c]))
                        data_sns['var_inter'].append(np.log(var_inter[c]))
                        data_sns['var_intra'].append(np.log(var_intra[c]))

        new_data = remove_outliers(data_sns, factors=['condition'], measure='separability')

        with open('separability_{}.csv'.format(data_phase), 'w') as f:  # You will need 'wb' mode in Python 2.x
            keys = new_data.keys()
            w = csv.DictWriter(f, keys)
            w.writeheader()
            for l in range(len(new_data['pid'])):
                row_to_write = {}
                for k in keys:
                    row_to_write[k] = new_data[k][l]
                w.writerow(row_to_write)

        df = pd.DataFrame(new_data)
        print('** one-way ANOVA')
        model = ols('separability ~ C(condition)', data=df).fit()
        print(sm.stats.anova_lm(model, typ=1)) 
        print('** pairwise t-tests')
        for ci, c in enumerate(np.unique(new_data['condition'])):
            for c2i, c2 in enumerate(np.unique(new_data['condition'])):
                if c2i > ci:
                    print('{} - {}\t'.format(c,c2), 
                            stats.ttest_ind(df[df['condition'] == c]['separability'], df[df['condition'] == c2]['separability']))

        sns.barplot(data=df, x='condition', y='separability')
        plt.title("separability")
        # plt.savefig('model_accuracy_testset={}_perclass={}_plot={}.png'.format(test_phase, perclass, type_of_plot))
        plt.savefig('separability_sep={}_phase={}_perclass={}_logscale={}.pdf'.format(
            type_of_sep, data_phase, perclass, log_scale))
        plt.show()

        plt.figure()
        plt.subplot(1,3,1)
        sns.barplot(data=df, x='condition', y='separability')
        plt.title('separability')
        plt.subplot(1,3,2)
        sns.barplot(data=df, x='condition', y='var_inter')
        plt.title('var inter')
        plt.subplot(1,3,3)
        sns.barplot(data=df, x='condition', y='var_intra')
        plt.title('var intra')
        plt.show()

    else:
        print("using models")
        data_sns = {
            'pid': [], 
            'condition': [], 
            'trial': [], 
            'separability': [], 
            'var_inter': [], 
            'var_intra': []}
        if perclass:
            data_sns['class'] = []

        for pid in data.keys():
            X = np.array(data[pid][data_phase]['x'])
            y = np.array(data[pid][data_phase]['y'])    

            batch_size = 49
            overlap_size = 1
            for trial in range(17, len(X), overlap_size):
                print(trial)
                inst_per_class = int(np.min([batch_size, trial]) / 8)
                idxes = []
                for c in np.unique(y):
                    idx = np.where(y[:trial] == c)[0]
                    if len(idx) < inst_per_class:
                        idxes.extend(idx)
                    else:
                        idxes.extend(idx[len(idx)-inst_per_class:])
                classifier = models[pid]['training'][trial]
                if type_of_sep == 'lda':
                    sep_perclass, var_inter, var_intra = separability(
                        classifier.transform(X[idxes, :]), 
                        y[idxes])
                elif type_of_sep == 'raw':
                    sep_perclass, var_inter, var_intra = separability(
                        X[idxes, :], 
                        y[idxes])

                if not perclass:
                    sep = np.mean([sep_perclass[c] for c in sep_perclass.keys()])
                    vinter = np.mean([var_inter[c] for c in var_inter.keys()])
                    vintra = np.mean([var_intra[c] for c in var_intra.keys()])
                    data_sns['pid'].append(pid)
                    data_sns['condition'].append(data[pid]['cond'])
                    data_sns['trial'].append(trial-16)
                    if not log_scale:
                        data_sns['separability'].append(sep)
                        data_sns['var_inter'].append(vinter)
                        data_sns['var_intra'].append(vintra)
                    else:
                        data_sns['separability'].append(np.log(sep))
                        data_sns['var_inter'].append(np.log(vinter))
                        data_sns['var_intra'].append(np.log(vintra))

                else:
                    for c in sep_perclass.keys():
                        data_sns['pid'].append(pid)
                        # data_sns['class'].append(c)
                        data_sns['class'].append(list(classifier.classes_).index(c))
                        data_sns['condition'].append(data[pid]['cond'])
                        data_sns['trial'].append(trial-16)
                        if not log_scale:
                            data_sns['separability'].append(sep_perclass[c])
                            data_sns['var_inter'].append(var_inter[c])
                            data_sns['var_intra'].append(var_intra[c])
                        else:
                            data_sns['separability'].append(np.log(sep_perclass[c]))
                            data_sns['var_inter'].append(np.log(var_inter[c]))
                            data_sns['var_intra'].append(np.log(var_intra[c]))
            
        new_data = remove_outliers(data_sns, factors = ['condition', 'trial'], measure='separability')

        with open('separability_along_training.csv', 'w') as f: 
            keys = new_data.keys()
            w = csv.DictWriter(f, keys)
            w.writeheader()
            for l in range(len(new_data['pid'])):
                row_to_write = {}
                for k in keys:
                    row_to_write[k] = new_data[k][l]
                w.writerow(row_to_write)

        df = pd.DataFrame(new_data)
        # for t in sorted(np.unique(new_data['trial'])):
        #     fval, pval = stats.f_oneway(
        #         df[(df['trial'] == t) & (df['condition'] == 'NUC')]['separability'],
        #         df[(df['trial'] == t) & (df['condition'] == 'NUCS')]['separability'],
        #         df[(df['trial'] == t) & (df['condition'] == 'UC')]['separability'])
        #     print(t, '\t', pval)
        sns.lineplot(data=df, x='trial', y='separability', hue='condition')
        # plt.title('retrained models')
        plt.savefig('separability_along_training_sep={}_phase={}_perclass={}_logscale={}.pdf'.format(
            type_of_sep, data_phase, perclass, log_scale))
        plt.show()

        # plt.figure()
        # plt.subplot(1,3,1)
        # sns.lineplot(data=df, x='trial', y='separability', hue='condition')
        # plt.title('separability')
        # plt.subplot(1,3,2)
        # sns.lineplot(data=df, x='trial', y='var_inter', hue='condition')
        # plt.title('var inter')
        # plt.subplot(1,3,3)
        # sns.lineplot(data=df, x='trial', y='var_intra', hue='condition')
        # plt.title('var intra')
        # plt.show()

        # df1 = df[(df['trial'] == 1)]
        # sns.violinplot(data=df1, x='condition', y='separability')

        # plt.show()

def compute_consistency(data, 
                         models, 
                         data_phase, 
                         model_t, 
                         perclass,
                         log_scale):
    # depending on whether we consider per-class accuracy or mean accuracy over classes
    if model_t != None:
        data_sns = {
            'pid': [], 
            'condition': [], 
            'consistency': []}
        if perclass:
            data_sns['class'] = []
        
        for pid in data.keys():
            X = np.array(data[pid][data_phase]['x'])
            y = np.array(data[pid][data_phase]['y'])
            if data_phase == 'training':
                X = X[:model_t,:]
                y = y[:model_t]
            classifier = models[pid]['training'][model_t]
            consist_perclass = consistency(X, y)

            if not perclass:
                consist = np.mean([consist_perclass[c] for c in const_perclass.keys()])
                data_sns['pid'].append(pid)
                data_sns['condition'].append(data[pid]['cond'])
                if not log_scale:
                    data_sns['consistency'].append(consist)
                else:
                    data_sns['consistency'].append(np.log(consist))
            else:
                for c in consist_perclass.keys():
                    data_sns['pid'].append(pid)
                    data_sns['class'].append(list(classifier.classes_).index(c))
                    data_sns['condition'].append(data[pid]['cond'])
                    if not log_scale:
                        data_sns['consistency'].append(consist_perclass[c])
                    else:
                        data_sns['consistency'].append(np.log(consist_perclass[c]))

        new_data = remove_outliers(data_sns, factors=['condition'], measure='consistency')

        with open('consistency_{}.csv'.format(data_phase), 'w') as f:  # You will need 'wb' mode in Python 2.x
            keys = new_data.keys()
            w = csv.DictWriter(f, keys)
            w.writeheader()
            for l in range(len(new_data['pid'])):
                row_to_write = {}
                for k in keys:
                    row_to_write[k] = new_data[k][l]
                w.writerow(row_to_write)

        df = pd.DataFrame(new_data)
        print('** one-way ANOVA')
        model = ols('consistency ~ C(condition)', data=df).fit()
        print(sm.stats.anova_lm(model, typ=1)) 
        print('** pairwise t-tests')
        for ci, c in enumerate(np.unique(new_data['condition'])):
            for c2i, c2 in enumerate(np.unique(new_data['condition'])):
                if c2i > ci:
                    print('{} - {}\t'.format(c,c2), 
                            stats.ttest_ind(df[df['condition'] == c]['consistency'], df[df['condition'] == c2]['consistency']))

        sns.barplot(data=df, x='condition', y='consistency')
        plt.title("consistency")
        # plt.savefig('model_accuracy_testset={}_perclass={}_plot={}.png'.format(test_phase, perclass, type_of_plot))
        plt.show()

    else:
        data_sns = {
            'pid': [], 
            'condition': [], 
            'trial': [], 
            'consistency': []}
        if perclass:
            data_sns['class'] = []

        for pid in data.keys():
            X = np.array(data[pid][data_phase]['x'])
            y = np.array(data[pid][data_phase]['y'])

            # batch_size = 17
            # for trial in range(batch_size, len(X), 1):
            #     inst_per_class = int(batch_size / 8)
            #     idxes = []
            #     for c in np.unique(y):
            #         idx = np.where(y[:trial] == c)[0]
            #         if len(idx) < inst_per_class:
            #             idxes.extend(idx)
            #         else:
            #             idxes.extend(idx[len(idx)-inst_per_class:])
            batch_size = 49
            overlap_size = 1
            for trial in range(17, len(X), overlap_size):
                inst_per_class = int(np.min([batch_size, trial]) / 8)
                idxes = []
                for c in np.unique(y):
                    idx = np.where(y[:trial] == c)[0]
                    if len(idx) < inst_per_class:
                        idxes.extend(idx)
                    else:
                        idxes.extend(idx[len(idx)-inst_per_class:])
                classifier = models[pid]['training'][trial]
                consist_perclass = consistency(
                    X[idxes,:], 
                    y[idxes])

                if not perclass:
                    consist = np.mean([consist_perclass[c] for c in consist_perclass.keys()])
                    data_sns['pid'].append(pid)
                    data_sns['condition'].append(data[pid]['cond'])
                    data_sns['trial'].append(trial-16)
                    if not log_scale:
                        data_sns['consistency'].append(consist)
                    else:
                        data_sns['consistency'].append(np.log(consist))

                else:
                    for c in consist_perclass.keys():
                        data_sns['pid'].append(pid)
                        # data_sns['class'].append(c)
                        data_sns['class'].append(list(classifier.classes_).index(c))
                        data_sns['condition'].append(data[pid]['cond'])
                        data_sns['trial'].append(trial-16)
                        if not log_scale:
                            data_sns['consistency'].append(consist_perclass[c])
                        else:
                            data_sns['consistency'].append(np.log(consist_perclass[c]))
            
        new_data = remove_outliers(data_sns, factors = ['condition', 'trial'], measure='consistency')

        with open('consistency_along_training.csv', 'w') as f: 
            keys = new_data.keys()
            w = csv.DictWriter(f, keys)
            w.writeheader()
            for l in range(len(new_data['pid'])):
                row_to_write = {}
                for k in keys:
                    row_to_write[k] = new_data[k][l]
                w.writerow(row_to_write)

        df = pd.DataFrame(new_data)
        sns.lineplot(data=df, x='trial', y='consistency', hue='condition')
        # plt.title('retrained models')
        # plt.savefig('separability_along_training_perclass={}_logscale={}.png'.format(
            # perclass, log_scale))
        plt.show()

def correlations(data, models, model_t, perclass):
    data_sns = {'pid':[], 'condition': [], 'accuracy': [], 'separability': []}
    for pid in data.keys():    
        classifier = models[pid]['training'][model_t]
        if not perclass:
            acc = classifier.score(
                np.array(data[pid]['posttest']['x']),
                np.array(data[pid]['posttest']['y']))
            # X = np.array(data[pid]['training']['x'])
            # y = np.array(data[pid]['training']['y'])
            X = np.array(data[pid]['posttest']['x'])
            y = np.array(data[pid]['posttest']['y'])
            # sep_pc = separability(classifier.transform(X), y)
            sep_pc = separability(X, y)
            sep = np.mean([sep_pc[c] for c in sep_pc.keys()])
            data_sns['pid'].append(pid)
            data_sns['condition'].append(data[pid]['cond'])
            data_sns['accuracy'].append(acc)
            data_sns['separability'].append(np.log(sep))
        else:
            # print(sorted(models[pid]['training'].keys()))
            classifier = models[pid]['training'][17]
            cm = confusion_matrix(
                np.array(data[pid]['posttest']['y']), 
                classifier.predict(np.array(data[pid]['posttest']['x'])), 
                labels=classifier.classes_)
            accs_start_training = {}
            for r in range(len(cm)):
                total = 0
                for c in range(len(cm)):
                    if r == c:
                        diag = cm[r,c]
                    total += cm[r,c]
                accs_start_training[classifier.classes_[r]] = diag / total
            
            classifier = models[pid]['training'][120]
            cm = confusion_matrix(
                np.array(data[pid]['posttest']['y']), 
                classifier.predict(np.array(data[pid]['posttest']['x'])), 
                labels=classifier.classes_)
            accs_end_training = {}
            for r in range(len(cm)):
                total = 0
                for c in range(len(cm)):
                    if r == c:
                        diag = cm[r,c]
                    total += cm[r,c]
                accs_end_training[classifier.classes_[r]] = diag / total
            
            # X = np.array(data[pid]['training']['x'])
            # y = np.array(data[pid]['training']['y'])
            X = np.array(data[pid]['posttest']['x'])
            y = np.array(data[pid]['posttest']['y'])
            # sep_pc = separability(classifier.transform(X), y)
            sep_pc = consistency(X, y)
            sep = np.mean([sep_pc[c] for c in sep_pc.keys()])

            for c in sep_pc.keys():
                # data_sns['pid'].append(pid)
                # data_sns['condition'].append(data[pid]['cond'])
                # data_sns['accuracy'].append(acc)
                # data_sns['separability'].append(np.log(sep))
                data_sns['pid'].append(pid)
                data_sns['condition'].append(data[pid]['cond'])
                data_sns['accuracy'].append(accs_end_training[c] - accs_start_training[c])
                data_sns['separability'].append(np.log(sep_pc[c]))

    for c in np.unique(data_sns['condition']):
        idx = np.where(np.array(data_sns['condition']) == c)[0]
        accs = np.array(data_sns['accuracy'])[idx]
        seps = np.array(data_sns['separability'])[idx]
        slope, intercept, r, p, se = stats.linregress(accs, seps)
        print(c, r, p)
        plt.plot(accs, seps, 'o', label='original data')
        plt.plot(accs, intercept + slope*accs, 'r', label='fitted line')
        plt.xlabel('acc')
        plt.ylabel('sep')
        plt.show()


def inspect_init_separability(data, models, perclass, log_scale):
    if not perclass:
        data_sns = {'pid': [], 'condition': [], 'separability': []}
    else:
        data_sns = {'pid': [], 'condition': [], 'class': [], 'separability': []}

    for pid in data.keys():
        print(pid)

        # if pid != 23:
        
        X = np.array(data[pid]['training']['x'])
        y = np.array(data[pid]['training']['y'])

        trial = 16
        classifier = models[pid]['init']
        X_ = classifier.transform(X[:trial, :])
        y_ = y[:trial]
        
        sep_perclass = separability(
                classifier.transform(X[:trial, :]), 
                y[:trial])

        if not perclass:
            sep = np.mean([sep_perclass[c] for c in sep_perclass.keys()])
            data_sns['pid'].append(pid)
            data_sns['condition'].append(data[pid]['cond'])
            if not log_scale:
                data_sns['separability'].append(sep)
            else:
                data_sns['separability'].append(np.log(sep))

        else:
            for c in sep_perclass.keys():
                data_sns['pid'].append(pid)
                data_sns['class'].append(c)
                data_sns['condition'].append(data[pid]['cond'])
                if not log_scale:
                    data_sns['separability'].append(sep_perclass[c])
                else:
                    data_sns['separability'].append(np.log(sep_perclass[c]))
        
    new_data = remove_outliers(
        data_sns, 
        factors = ['condition'], 
        measure='separability')

    with open('separability_init.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        keys = new_data.keys()
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for l in range(len(new_data['pid'])):
            row_to_write = {}
            for k in keys:
                row_to_write[k] = new_data[k][l]
            w.writerow(row_to_write)

    df = pd.DataFrame(new_data)
    sns.catplot(data=df, x="condition", y="separability")
    plt.show()

def improvement_in_separability(data, models):

    confusion_mat = {}

    for pid in data.keys():

        cond = data[pid]['cond']

        X = np.array(data[pid]['training']['x'])
        y = np.array(data[pid]['training']['y'])
        classifier = models[pid]['training'][120]

        classes_ = list(classifier.classes_)
        
        if cond not in confusion_mat.keys():
            confusion_mat[cond] = {}
        
        confusion_mat[cond][pid] = []

        all_data = {}
        centroids = {}
        for i in range(17):
            if y[i] not in all_data.keys():
                all_data[y[i]] = []
            all_data[y[i]].append(X[i,:])
        
        for k in all_data.keys():
            centroids[k] = np.mean(all_data[k], axis=0)
        
        confusion_mat_tm1 = np.zeros((len(classes_), len(classes_)))
        for k1 in centroids.keys():
            for k2 in centroids.keys():
                # if k1 != k2:
                confusion_mat_tm1[classes_.index(k1), classes_.index(k2)] = np.sqrt(
                    np.sum(np.power(centroids[k1] - centroids[k2], 2)))
        
        confusion_mat_t = np.copy(confusion_mat_tm1)
        for i in range(17, len(X)):
            all_data[y[i]].append(X[i,:])
            centroids[y[i]] = np.mean(all_data[y[i]], axis=0)

            for ke in centroids.keys():
                if ke != y[i]:
                    confusion_mat_t[classes_.index(y[i]), classes_.index(ke)] = np.sqrt(
                        np.sum(np.power(centroids[ke] - centroids[y[i]], 2)))
                    confusion_mat_t[classes_.index(ke), classes_.index(y[i])] = np.sqrt(
                        np.sum(np.power(centroids[ke] - centroids[y[i]], 2)))
                    # print(pid, ke, y[i], np.sqrt(
                    #     np.sum(np.power(centroids[ke] - centroids[y[i]], 2))))
            # print(y[i], classes_.index(y[i]), confusion_mat_t)
            confusion_mat[cond][pid].append(np.copy(confusion_mat_t))
            # confusion_mat_tm1 = np.copy(confusion_mat_t)

    mean_confusion_mat = {}
    for c in confusion_mat.keys():
        mean_confusion_mat[c] = []
        # print(confusion_mat[c][list(confusion_mat[c].keys())[0]])
        for i in range(len(confusion_mat[c][list(confusion_mat[c].keys())[0]])):
            c_ = np.zeros((len(classes_), len(classes_)))
            for pid in confusion_mat[c].keys():
                c_ = (c_ + confusion_mat[c][pid][i]) / len(confusion_mat[c].keys())
            mean_confusion_mat[c].append(c_)
            # print(c, i, mean_confusion_mat[c][-1])
    
    print(classes_)
    plt.figure(figsize=(15,6))
    for ci, c in enumerate(confusion_mat.keys()):
        ax = plt.subplot(1,3,ci+1)
        
        diff_conf = mean_confusion_mat[c][-1] - mean_confusion_mat[c][0]
        # print(c, np.min(diff_conf), np.max(diff_conf))
        im = plt.imshow(diff_conf, vmin=-0.8, vmax=0.87)
        # for i in range(len(diff_conf)):
        #     for j in range(len(diff_conf)):
        #         text = ax.text(j, i, diff_conf[i, j],
        #                     ha="center", va="center", color="w")
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.title(c)
    plt.show()

    
    for cond in confusion_mat.keys():
        toplot = [mean_confusion_mat[cond][i][0,7] for i in range(len(mean_confusion_mat[cond]))]
        # print(len(mean_confusion_mat[cond]))
        plt.plot(toplot, label=cond)
    plt.legend()
    plt.show()

def compute_pos_neg_metric(data, 
                           models, 
                           train_phase, 
                           model_t, 
                           perclass,
                           metric,
                           type_of_plot):
    
    # depending if we consider per-class accuracy or mean accuracy over classes
    if not perclass: 
        data_sns = {'pid': [], 'condition': [], 'type': []}
    else: 
        data_sns = {'pid': [], 'condition': [], 'class': [], 'type': []}
    data_sns['rate'] = []

    for pid in data.keys():
        
        if models == None:
            classifier = LinearDiscriminantAnalysis()
            X = np.array(data[pid][train_phase]['x'])[:model_t,:]
            y = np.array(data[pid][train_phase]['y'])[:model_t]
            classifier.fit(X, y)
        else:
            classifier = models[pid]['training'][model_t]
        
        condition = data[pid]['cond']
        # if metric == 'accuracy':
        for t in ['positive', 'negative']:
            if not perclass:
                data_sns['pid'].append(pid)
                data_sns['condition'].append(condition)
                data_sns['type'].append(t)
                data_sns['rate'].append(
                    classifier.score(
                        np.array(data[pid][t]['x']),
                        np.array(data[pid][t]['y'])))
            else:
                tpr, tp, p = perclass_accuracy(
                    np.array(data[pid][t]['y']),
                    classifier.predict(np.array(data[pid][t]['x'])), 
                    classifier.classes_
                )
                for k in tpr.keys():
                    data_sns['pid'].append(pid)
                    data_sns['condition'].append(data[pid]['cond'])
                    data_sns['class'].append(k)
                    if t == 'positive':
                        data_sns['type'].append('True Positive Rate')
                        data_sns['rate'].append(tpr[k])
                    else:
                        data_sns['type'].append('True Negative Rate')
                        data_sns['rate'].append((p[k] - tp[k])/p[k])
                        #data_sns['rate'].append(tn[k]/p[k])
        # else:
        #     fscore = {}
        #     false_negatives = {}
        #     true_positives = {}
        #     false_positives = {}
        #     for t in ['positive', 'negative']:
        #         cm = confusion_matrix(
        #             y_true, 
        #             y_pred, labels=classes)
        #         if t == 'positive':
        #             for r in range(len(cm)):
        #                 total = 0
        #                 for c in range(len(cm)):
        #                     if r == c:
        #                         true_positives[r] = cm[r,c]
        #                     total += cm[r,c]
        #                 true_negatives[r] = true_positives[r] / total
        #                 # if total != 0:
        #                 #     perclass_accuracy_[r] = diag / total
        #                 # else:
        #                 #     perclass_accuracy_[r] = 1.0
        #         else: 
        #             for r in range(len(cm)):
        #                 for c in range(len(cm)):
        #                     if r == c:
        #                         false_positives[r] = cm[r,c]
        #     for k in true_positives.keys():
        #         fscore[k] = 2 * true_positives[k] / (2 * true_positives[k] + false_positives[k] + false_negatives[k])
        #     for k in perclass_acc.keys():
        #         data_sns['pid'].append(pid)
        #         data_sns['condition'].append(data[pid]['cond'])
        #         data_sns['class'].append(k)
        #         data_sns['type'].append('posneg')
        #         data_sns['accuracy'].append(fscore[k])


    conds_tab = ['NUC', 'NUCS', 'UC']
    type_tab = ['positive', 'negative']
    with open('pos_neg.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        keys = data_sns.keys()
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for l in range(len(data_sns['pid'])):
            row_to_write = {}
            for k in keys:
                # if k == 'condition':
                #     row_to_write[k] = conds_tab.index(data_sns[k][l])
                # elif k == 'type':
                #     row_to_write[k] = type_tab.index(data_sns[k][l])
                # else:
                #     row_to_write[k] = data_sns[k][l]
                row_to_write[k] = data_sns[k][l]
            # if row_to_write['type'] == 0:
            w.writerow(row_to_write)
            



    df = pd.DataFrame(data_sns)
    # print('** two-way ANOVA')
    # model = ols('accuracy ~ C(condition) + C(type) + C(condition):C(type)', data=df).fit()
    # print(sm.stats.anova_lm(model, typ=2))    
    # print('** pairwise test')
    # for t in ['positive', 'negative']:
    #     for ci, c in enumerate(np.unique(data_sns['condition'])):
    #         for c2i, c2 in enumerate(np.unique(data_sns['condition'])):
    #             if c2i > ci:
    #                 print('{} | {} - {}\t'.format(t, c,c2), 
    #                       stats.ttest_ind(
    #                         df[(df['type'] == t) & (df['condition'] == c)]['accuracy'], 
    #                         df[(df['type'] == t) & (df['condition'] == c2)]['accuracy']))
    # if type_of_plot== 'barplot':
    sns.barplot(data=df, x='condition', y='rate', hue='type')
    # elif type_of_plot== 'violinplot':
    # sns.violinplot(data=df, x='condition', y='accuracy', hue='type', inner="points")
    # plt.title("retrained models - anova's pvalue={:.3f}".format(pval))
    plt.savefig('pos-neg-analysis_plot={}_perclass={}.png'.format(type_of_plot, perclass))
    plt.show()

def questionnaires(file, data, models, model_id=120):
    maps = {'NUC': 'RC', 'UC': 'UCC', 'NUCS': 'GSC'}
    df = pd.read_csv(file, delimiter=',') 
    cols= {'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Palm Up]' : 'Palm Up',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Palm Down]' : 'Palm Down',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Rest Hand]' : 'Rest Hand',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Open Hand]' : 'Open Hand',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Close Hand]' : 'Close Hand',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Open Pinch]' : 'Open Pinch',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Close Pinch ]' : 'Close Pinch',
          'Selon votre expérience, après avoir appris au système à reconnaître les 8 gestes, avec quelle précision est-ce que le système arrivera à bien reconnaître chaque geste\u202f? \r\n(0% - le système ne reconnaît jamais le geste à \r\n100% le système reconnaît toujours le geste) [Point Index]' : 'Point Index'}
    df.rename(columns=cols, inplace=True)
    df_dict = df.to_dict()
    # print(df_dict)
    participant_answer = {'UCC':{}, 'RC':{}, 'GSC':{}}
    
    gesture_names = []
    new_cond = {'UC':'UCC','NUC':'RC','NUCS':'GSC'}
    data_sns = {'pid':[],'curriculum':[],'perceived':[],'actual':[],'class':[]}
    d_temp = {'UCC':{},'RC':{},'GSC':{}}
    for i in range(len(df_dict.keys())):
        if(i >= 6 and i < 14):
            gesture_names.append(list(df_dict.keys())[i])
    
    
    for index in df_dict['PID'].keys():

        pid = df_dict['PID'][index]
        if(pid in data.keys()):

            cond =  new_cond[df_dict['curriculum'][index]]
            classifier = models[pid]['training'][model_id] 
            accuracy, tp, p = perclass_accuracy(
                data[pid]['posttest']['y'],
                classifier.predict(np.array(data[pid]['posttest']['x'])),
                classifier.classes_)
            
            # print(pid, cond, accuracy)

            d_temp[cond][pid] = {}
            for k in accuracy.keys():
                d_temp[cond][pid][classifier.classes_[int(k)]] = accuracy[k]
            # temp_list = [
            #         d_temp[cond][pid][name] for name in sorted(d_temp[cond][pid].keys())]
            # temp_list = stats.zscore(temp_list)
            # for i, name in enumerate(sorted(d_temp[cond][pid].keys())):
            #     d_temp[cond][pid][name] = temp_list[i]

            # temp_list = [d_temp[cond][pid][name] for name in sorted(d_temp[cond][pid].keys())]
            # print(d_temp[cond][pid])
            # for i, name in enumerate(sorted(d_temp[cond][pid].keys())):
            #     d_temp[cond][pid][name] = temp_list[i]
            # print(d_temp[cond][pid])

            if((pid not in participant_answer[cond].keys()) #and (pid in accuracy[cond].keys())
            ):
                participant_answer[cond][pid] = {} 
                for name in gesture_names:
                    if(name not in participant_answer[cond][pid].keys()):
                        participant_answer[cond][pid][name] = int(df_dict[name][index][0:-1])/100
                        # print(pid, cond, name, participant_answer[cond][pid][name])
                # temp_list = [
                #     participant_answer[cond][pid][name] for name in sorted(participant_answer[cond][pid].keys())]
                # temp_list = stats.zscore(temp_list)
                # for i, name in enumerate(sorted(participant_answer[cond][pid].keys())):
                #     participant_answer[cond][pid][name] = temp_list[i]
                # print(cond, pid, name, participant_answer[cond][pid][name], d_temp[cond][pid][name])
       
    correlations = {'UCC': [], 'RC': [], 'GSC': []}
    for curriculum in participant_answer.keys():
        for pid in participant_answer[curriculum].keys():  
            answer = []
            model_acc = []
            for gesture in participant_answer[curriculum][pid].keys():
                data_sns['pid'].append(pid)
                data_sns['curriculum'].append(curriculum)
                data_sns['perceived'].append(participant_answer[curriculum][pid][gesture])
                data_sns['actual'].append(d_temp[curriculum][pid][gesture])
                data_sns['class'].append(gesture)

    # print(correlations)
    for c in np.unique(data_sns['curriculum']):
        idx = np.where(np.array(data_sns['curriculum']) == c)[0]
        r,p= stats.pearsonr(
            np.array(data_sns['perceived'])[idx],
            np.array(data_sns['actual'])[idx])
        # res = stats.spearmanr(
        #     np.array(data_sns['perceived'])[idx],
        #     np.array(data_sns['actual'])[idx])
        ax = plt.subplot(1,1,1)
        sns.regplot( x = np.array(data_sns['perceived'])[idx], y = np.array(data_sns['actual'])[idx], color=".3")
        # plt.title(c+':'+' corr={}'.format(str(round(r,3)))+' p={}'.format(str(round(p,5))))
        plt.xlabel('Perceived accuracy (zscore)', fontsize=13)
        plt.ylabel('Actual accuracy (zscore)', fontsize=13)
        plt.title(c, fontsize=15)
        ax.text(1.55, 1.0, "r={:.2f}\np={:.3f}".format(r,p), 
                fontsize = 13,          # Size
                fontstyle = "oblique",  # Style
                color = "red",          # Color
                ha = "center", # Horizontal alignment
                va = "center")
        plt.xlim([-3,2])
        plt.ylim([-3,2.5])
        plt.savefig('correlation_questionnaire_{}.pdf'.format(c))
        plt.show()

        print(c, r, p)


    with open('correlation.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
            keys = data_sns.keys()
            w = csv.DictWriter(f, keys)
            w.writeheader()
            for l in range(len(data_sns['pid'])):
                row_to_write = {}
                for k in keys:
                    row_to_write[k] = data_sns[k][l]
                w.writerow(row_to_write)

def questionnaire(filename):
    subj_confus = {}
    with open(filename) as csv_file:
        data_csv=pd.read_csv(
            csv_file, 
            delimiter=',',
            names=["Timestamp", "PID", "Age", "Sexe", "ML","Robotics", "Palm Up", "Palm Down", "Rest Hand", "Open Hand", "Close Hand", "Open Pinch", "Close Pinch", "Point Index"])
        
        for p in np.unique(data_csv['PID']):
            print(p)
            ss = []
            for g in ["Palm Up", "Palm Down", "Rest Hand", "Open Hand", "Close Hand", "Open Pinch", "Close Pinch", "Point Index"]:
                print(data_csv[data_csv['PID'] == p][g])
                ss.append(int(data_csv[data_csv['PID'] == p][g][:-1]))
            data_csv[p] = ss
        print(data_csv)
        # for k in data_csv.columns[6:]:
        #     gesture = k.split('[')[-1][:-1]
        # for k in range(2):
        #     print(data_csv.loc[k])
        # # for row in csv_read:
        # #     print(row)

def learning_rate(data,models):
    data_sns = {'pid': [], 'condition': [], 'learning rate': [], 'intercept': []}
    data_sns2 = {'pid': [], 'condition': [], 'trial': [], 'accuracy': []}
    for pid in data.keys():
        
        acc = []
        X = np.array(data[pid]['training']['x'])
        y = np.array(data[pid]['training']['y'])

        for trial in range(17,len(X)):        
            classifier = models[pid]['training'][trial]
            acc.append(classifier.score(
                        np.array(data[pid]['posttest']['x']),
                        np.array(data[pid]['posttest']['y'])))
            
            data_sns2['pid'].append(pid)
            data_sns2['condition'].append(data[pid]['cond'])
            data_sns2['trial'].append(trial)
            data_sns2['accuracy'].append(classifier.score(
                        np.array(data[pid]['posttest']['x']),
                        np.array(data[pid]['posttest']['y'])))
            # print(classifier.score(
            #         np.array(data[pid]['posttest']['x']),
            #         np.array(data[pid]['posttest']['y'])))

        data_sns['pid'].append(pid)
        data_sns['condition'].append(data[pid]['cond'])
        print(acc)
        slope, intercept, r, p, std_err = stats.linregress(
            np.log(np.arange(17, len(X))), np.log(np.array(acc)))
        data_sns['learning rate'].append(slope)
        data_sns['intercept'].append(intercept)
    
    # for k in np.unique(data_sns['condition']):
    #     idx = np.where(np.array(data_sns2['condition']) == k)[0]
    #     temp_arr = np.array(data_sns2['trial'])[idx]
    #     temp_arr2 = np.array(data_sns2['accuracy'])[idx]
    #     acc_ = []
    #     indexes = []
    #     for i in np.unique(temp_arr):
    #         idx2 = np.where(temp_arr == i)[0]
    #         acc_.append(np.mean(temp_arr2[idx2]))
    #         indexes.append(i)
    #     indexes = np.array(indexes)
    #     acc_ = np.array(acc_)
    #     plt.plot(indexes, np.log(np.log(1.0/acc_)), 'o', label=k)
    #     slope, intercept, r, p, std_err = stats.linregress(indexes, np.log(np.log(1.0/acc_)))
    #     plt.plot(indexes, intercept + slope * indexes, '-')
    # plt.legend()
    # plt.show()

    # loga = s * x + b
    # a = exp(s * x + b) = exp(s * x) * exp(b) 

    # plt.figure()
    # for k in np.unique(data_sns['condition']):
    #     idx = np.where(np.array(data_sns['condition']) == k)[0]
    #     m_lr = np.mean(np.array(data_sns['lr'])[idx])
    #     intercept = np.mean(np.array(data_sns['intercept'])[idx])
    #     y = np.log(intercept + np.arange(0, len(X)-17) * m_lr)
    #     plt.plot(np.arange(0, len(X)-17), y)
    # plt.plot(np.arange(0, len(X)-17), np.log(intercept + np.arange(0, len(X)-17) * 1.8))
    # plt.show()

    new_data = remove_outliers(data_sns, factors=['condition'], measure='learning rate')

    with open('lr.csv', 'w') as f: 
        keys = new_data.keys()
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for l in range(len(new_data['pid'])):
            row_to_write = {}
            for k in keys:
                row_to_write[k] = new_data[k][l]
            w.writerow(row_to_write)

    df = pd.DataFrame(new_data)
    # model = ols('lr ~ C(condition)', data=df).fit()
    # print(sm.stats.anova_lm(model, typ=1)) 

    sns.barplot(data=df, x='condition', y='learning rate',palette='Set2',capsize=.1,order=['NUC', 'UC', 'NUCS'])
    # plt.title("retrained models - anova's pvalue={:.3f}".format(pval))
    # plt.savefig('model_accuracy_testset={}_perclass={}_plot={}.png'.format(test_phase, perclass, type_of_plot))
    plt.show()

def plot_csv(filename):
    accs = pd.read_csv(filename)
    sns.barplot(data=accs, x='condition', y='accuracy')
    plt.show()

def plot_distribution(data):
    y = {}
    #print(data)
    for pid in data.keys():
        for val in np.array(data[pid]['training']['y']):
            if(val not in y.keys()):
                y[val] = 0
            y[val] += 1
    #print(y)
    dict_of_dicts ={'UC':{},'NUC':{},'NUCS':{}}
    for cond in ['UC','NUC','NUCS']:
        for pid in data.keys():  
            if data[pid]['cond'] == cond:
                #storing gestures in ascending order in dictionary
                dict_of_dicts[cond]= y
        
    for cond in dict_of_dicts.keys():
        print(cond, ':' ,dict_of_dicts[cond])

if __name__ == "__main__":
    # folder path
    dir_path = 'data'
    data, models = load_data(dir_path)
    # plot_distribution(data)
    # sanity_check(data)
    learning_rate(data, models)
    # # Accuarcy analysis
    # # -----------------
    # print('\n[Accuracy analysis]')
    # accuracy_analysis_config = {
    #     "train_phase": 'training', 
    #     "test_phase": 'posttest',
    #     "model_t": None,
    #     "perclass": True,
    #     "offset": False,
    #     "type_of_plot": 'barplot'
    # }
    # print('** test config\n', accuracy_analysis_config)
    # compute_clf_accuracy(
    #     data, 
    #     models=models,
    #     **accuracy_analysis_config)

    # check_uc_acc(data, models)
    # check_init_acc(data,models)

    # CORRELATIONS
    # correlations(data, models, 120, perclass=True)


    # # CONFUSION
    # print('\n[Accuracy analysis]')
    # accuracy_analysis_config = {
    #     "train_phase": 'training', 
    #     "test_phase": 'posttest',
    #     "model_t": 120,
    #     "perclass": False,
    # }
    # print('** test config\n', accuracy_analysis_config)
    # compute_confusion(
    #     data, 
    #     models=models,
    #     **accuracy_analysis_config)

    # learning_rate(data, models)
    
    # # Mental model analysis
    # # ---------------------
    # print('\n[Mental model: positive-negative]')
    # mental_model_pos_neg = {
    #     "train_phase": 'training', 
    #     "model_t": 120,
    #     "perclass": True,
    #     "metric": "rate",
    #     "type_of_plot": 'violinplot'
    # }
    # print('** test config\n', mental_model_pos_neg)
    # compute_pos_neg_metric(
    #     data, 
    #     models=models,
    #     **mental_model_pos_neg)

    # questionnaires(
    #     'Questionnaire post entrainement.csv', 
    #     data, 
    #     models=models, 
    #     model_id=120)

    # # Separability analysis
    # # ---------------------
    # print('\n[Separability analysis]')
    # sep_analysis = {
    #     "type_of_sep": 'lda',
    #     "data_phase": 'training', 
    #     "model_t": None,
    #     "perclass": True,
    #     "log_scale": True
    # }
    # print('** test config\n', sep_analysis)
    # compute_separability(data, 
    #     models=models,
    #     **sep_analysis)
    
    # improvement_in_separability(data, models)

    # # Consistency analysis
    # # ---------------------
    # print('\n[Consistency analysis]')
    # sep_analysis = {
    #     "data_phase": 'posttest', 
    #     "model_t": 120,
    #     "perclass": True,
    #     "log_scale": True
    # }
    # print('** test config\n', sep_analysis)
    # compute_consistency(data, 
    #     models=models,
    #     **sep_analysis)


    # inspect_init_separability(data, 
    #     models=models, 
    #     perclass=True, 
    #     log_scale=True)

    # # plot_csv('accuracies.csv')
    # data__ = {'pid':[], 'condition':[], 'class':[], 'type':[], 'accuracy':[]}
    # pos_neg = pd.read_csv('pos_neg.csv').to_dict()
    # acc = pd.read_csv('accuracies.csv').to_dict()
    
    # # print(pos_neg['pid'], pos_neg['pid'].keys())
    # for l in pos_neg['pid'].keys():
    #     # print(acc['pid'][l])
    #     data__['pid'].append(pos_neg['pid'][l])
    #     data__['condition'].append(pos_neg['condition'][l])
    #     data__['class'].append(pos_neg['class'][l])
    #     data__['type'].append(pos_neg['type'][l])
    #     data__['accuracy'].append(pos_neg['accuracy'][l])
    # for l in acc['pid'].keys():
    #     data__['pid'].append(acc['pid'][l])
    #     data__['condition'].append(acc['condition'][l])
    #     data__['class'].append(acc['class'][l])
    #     data__['type'].append('final')
    #     data__['accuracy'].append(acc['accuracy'][l])
    # df = pd.DataFrame(data__)
    # df.to_csv('pos_neg_final.csv', index=False)
