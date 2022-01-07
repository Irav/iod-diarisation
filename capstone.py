
import librosa
import librosa.display
import numpy as np
import pandas as pd
import math
import os
import sys
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn import preprocessing
from pydub import AudioSegment
class audio:
    def __init__(self):
        pass

    def data_gen(path, padsize, sr=16000,names_dict=None, label=None, flag=False, binarizer=None):
        #define n of features
        n_mfcc = 15
        #load
        audio, sr = librosa.load(path, sr=sr, mono=True)
        
        #features and deltas
        mfcc = librosa.feature.mfcc(audio, sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        #final array
        array = np.vstack((mfcc,delta, delta2))

        #make sure all are the same size    
        if array.shape[1] < padsize:
            padding = padsize - array.shape[1]
            pad = np.zeros(((n_mfcc*3), padding))
            array = np.append(array,pad, axis=1)    
            
        array = preprocessing.scale(array)
        if flag==True:
            #append labels
            if binarizer==None:
                print('Error! When appending labels you must pass a fitted binarizer')
                sys.exit()
            res = int("".join(str(x) for x in binarizer.transform([label])[0]),2)
            res = int(math.log(res,2)) +1
            names_dict[res] = label
            lab = np.full((array.shape[1],1),res)
            array = np.append(array,lab.T, axis=0)
        
        return array.T, names_dict

    def feat_ext(path, sr=22050, n_mfcc=15, energy_power=1, window='hann', win_l=300, n_fft=2048):
        #Audio Load
        y, sr = librosa.load(path, sr=sr, mono=True) 
        #Features
        stft = librosa.stft(y=y, window=window, win_length=win_l)
        S,phase= librosa.magphase(stft)
        energy = librosa.feature.melspectrogram(y=y, sr=sr, power=energy_power)
        rms = librosa.feature.rms(y=y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        dmfcc = librosa.feature.delta(mfcc)
        ddmfcc = librosa.feature.delta(mfcc, order=2)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, win_length=win_l)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, win_length=win_l)
        
        output = {'audio':y,
                'sr':sr,
                'stft':stft,
                'S':S,
                'energy':energy,
                'rms':rms,
                'mfcc':mfcc,
                'dmfcc':dmfcc,
                'ddmfcc':ddmfcc,
                'contrast':contrast,
                'centroid':centroid
        }
        return output

    def spec_plotter(df,filename,plot='energy',sr=22050,save=False, out_path=None, show=True):
        #feature to be graphed
        graph = librosa.amplitude_to_db(df[plot], ref=np.max)
        #plotting
        fig, ax = plt.subplots(1,1, sharex=True, figsize=[18,5])
        print(filename,'\n')
        librosa.display.specshow(graph,y_axis='log', x_axis = 'frames',sr=sr,ax=ax)
        
        if save: utils.fig_save(filename, out_path)
        if show: plt.show()

    def mask_plotter(backdrop, label_df,filename, save=False, wave=False, mask_color='w',
     out_path=None, suffix='', show=True):
        sns.set_theme(style='white')
        fig, ax = plt.subplots(1,1, figsize=[18,5])
        print(filename)
        
        if wave==False:
            librosa.display.specshow(librosa.amplitude_to_db(backdrop, ref=np.max),
                                 y_axis='log', x_axis = 'frames', ax=ax)
        else:
            ax.plot(backdrop, color='b', linestyle='-')
            
        ax2 = ax.twinx()
        ax2.plot(label_df['l'], label='threshold', color=mask_color, linestyle='-')

        if save==True:
            if save: utils.fig_save(filename, out_path, suffix=suffix)
        if show: plt.show()
    
    def summer(old_df, idx=False):


        df = old_df.copy()
        df['f'] = df['l'].shift()
        df["cumsum"] = (df['f'] != df['l']).cumsum()
        df = df.groupby('cumsum').agg(['min','max'])
        if idx:
            df = df[['l','idx']]    
        else:
            df = df[['l','min','max']]
        
        df2 = pd.DataFrame()
        df2['l'] = df['l']['min']
        if idx:
            df2['min'] = df['idx']['min']
            df2['max'] = df['idx']['max']
        else:
            df2['min'] = df['min']['min']
            df2['max'] = df['max']['max']
            
        df2 = df2[df2['min'] != df2['max']]
        df2['dur'] = df2['max'] - df2['min']
        
        df2.reset_index(inplace=True)
        df2.drop('cumsum', axis=1,inplace=True)
        return df2
    
    def chunker(file_path,out_path,df,fR):
        #export chunks
        for x in range(df.shape[0]):
            newAudio = AudioSegment.from_wav(file_path)
            t1 = int((df.loc[x,'min']/fR)*1000)  #miliseconds
            t2 = int((df.loc[x,'max']/fR)*1000)
            newAudio = newAudio[t1:t2]
            file_target = out_path +'chunks/'+f'chunk_s{x:04d}.wav'
            newAudio.export(file_target, format="wav")

    def id_me(file_list, model,labels):
        #Grabbing MFCC features
        test_sets = {}
        test_dfs = {}
        files=[]
        winners=[]
        ids = []
        preds_list = []

        #load audio and prep data
        for file in file_list:
            #grab name
            name=file.split('/')[file.count('/')]
            name=name.split('.')[0]  
            
            #generate data
            test_sets[name], _ = audio.data_gen(file,2000,names_dict=None)
            
            #dataframe conversion (within dict)
            test_dfs[name] = pd.DataFrame(test_sets[name])
            test_dfs[name] = test_dfs[name][~(test_dfs[name] ==0).any(axis=1)]
            
        #Making predictions per dataframe
        for dfx in test_dfs.keys():
            #print('Finding predictions for file:', dfx)
            #make predictions
            preds = pd.Series(model.predict(test_dfs[dfx]))
            
            #count values
            bunch = preds.value_counts()
            classes_ = dict.fromkeys(set(preds))
            
            for class_ in classes_:
                classes_[class_] = bunch[class_] / bunch.sum()
                    
            #print(classes_)
            lorg = max(classes_.values()) 
            #print(lorg)
            id_ = list(classes_.keys())[list(classes_.values()).index(lorg)]
            ids.append(id_)
            winner = labels[id_]
            files.append(dfx)
            winners.append(winner)
            preds_list.append(classes_)

        df=pd.DataFrame()
        df['id'] = ids
        df['speaker'] = winners
        df['filename'] = files
        df['conf'] = preds_list
        return df


class utils:
    def __init__(self):
        pass

    def dir_check(dir):
            #check if /landscape exists
            if os.path.isdir(dir) == False :
                os.mkdir(dir)

    def fig_save(filename, out_path, format='png', suffix=''):
        if out_path == None:
                out_path = os.getcwd()

        full_path=out_path+filename+suffix+'.'+format
        #print(full_path)
        plt.savefig(full_path, format=format)

    def print_shape(filename, dict):
        a = dict  
        for array in a:
            try:  
                #print('%20s \ %5s --> %2s' % (file, array, a[array].shape))
                print(f'{filename} - {array:<8} {"-->":^5} {a[array].shape}')
            except:
                continue
        print('\n')  
        return
    
    def array_pad(array, target):
        if array.ndim ==1:
            if array.shape[0] <  target:
                padding = target - array.shape[0]
                pad = np.zeros(padding)
                array = np.append(array,pad, axis=0)
        else:
            if array.shape[1] < target:
                padding = target - array.shape[1]
                pad = np.zeros(((array.shape[0]), padding))
                array = np.append(array,pad, axis=1)
        return array

    def label_expander(prefix, df, ratio, i=0, o=1,error=0.3):
        #column expansion
        in_ = f'{prefix}_in'
        out_ = f'{prefix}_out'
        
        df[in_] = df[i].apply(lambda x: x*ratio)
        df[out_] = df[o].apply(lambda x: x*ratio)
        
        #target df init
        lab_df = pd.DataFrame()
        
        #label cleanup
        error = (error*ratio)
        for x in range(df.shape[0]):        
            '''
            Since the labels are seconds-based, an error is introduced that creates
            a shift on the mask signal, which needs to be brought back into phase
            with the audio signal.
            '''
            #error correction
            it = 1
            if x>0 and x < df.shape[0]-1:
                if df.loc[x,1]*ratio > (error*it):
                    df.loc[x+1,in_]=df.loc[x+1,in_]+error*it
                    df.loc[x+1,out_]=df.loc[x+1,out_]+error*it
                    it +=1   
            #label conversion    
            df.loc[x+1,in_] = df.loc[x,out_]+1

        df.dropna(inplace=True)
        
        #type conversion
        df[[2,in_,out_]] = df[[2,in_,out_]].astype(int)

        #label expansion
        b=np.empty(1)
        lab_df['l'] = 0
        for x in range(df.shape[0]):
            ii = df.loc[x,in_]
            oo = df.loc[x,out_]
            dd = oo - ii
            a = np.empty(int(dd))
            a.fill(df.loc[x,2])
            b = np.append(b,a.ravel())
        lab_df['l']=b
        #lab_df['l']=lab_df['l'].astype(int)
        
        return lab_df        

    def label_apply(filename,df,feature, label_df):
        #create df
        X = pd.DataFrame(df[feature].T)
        #append labels
        X['l'] = label_df.astype(int)
        #fill zero
        X['l'].fillna(0, inplace=True)
        #split labels off
        y = X['l'].astype(int)
        X.drop('l',axis=1,inplace=True)
        
        print(X.shape, y.shape) 

        return X,y

        #Notebook Backup
        # def label_apply(file,feature):
        # #create df
        # globals()[f'{file}_{feature}_Xtr'] = pd.DataFrame(globals()[file][feature].T)
        # #append labels
        # globals()[f'{file}_{feature}_Xtr']['l'] = globals()[f'{file}_labels']
        # #fill zero
        # globals()[f'{file}_{feature}_Xtr']['l'].fillna(0, inplace=True)
        # #split labels off
        # globals()[f'{file}_{feature}_ytr'] = globals()[f'{file}_{feature}_train']['l']
        # globals()[f'{file}_{feature}_Xtr'].drop('l',axis=1,inplace=True)
        
        # print(f'{file}_{feature}_Xtr')

    def sloper(list):
        serie = pd.Series(list)
        if serie.is_monotonic_decreasing:
            flag = -1
        elif serie.is_monotonic_increasing:
            flag = 1
        else:
            flag = 0
        return flag