import numpy as np
import matplotlib.pyplot as plt
from lstm_autoencoder import create_lstm_autoencoder
import pandas as pd
import keras
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
       # plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
           # plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()
if __name__ == "__main__":
    dataX=[]
    dataY=[]
    vmd = pd.read_csv('1.csv',encoding='gbk')
    toubu = vmd[vmd["bone"]=="ЙEМи"].sort_values('Motion')
    print(vmd)
    for idx, row in toubu.iterrows():
        rx = row["rx"]
        ry = row["ry"]
        rz = row["rz"]
        dataX.append([[rx,ry,rz]])
    X=np.array(dataX)
    x = (X-X.min())/(X.max()-X.min())
    print(x.shape)
    input_dim = x.shape[-1] # 13
    timesteps = x.shape[1] # 3
    batch_size = 1

    autoencoder = create_lstm_autoencoder(input_dim, 
        timesteps=timesteps, 
        batch_size=batch_size, 
        intermediate_dim=32,
        latent_dim=100,
        epsilon_std=2.)
    history = LossHistory()
    autoencoder.fit(x, x, epochs=200,batch_size=batch_size,
            verbose=1, 
            validation_data=(x, x),
            callbacks=[history])

    preds = np.zeros((584,1,3))
    preds[0:3,:,:] = x[0:3,:,:]
    for i in range(584):
        preds[i+1:i+2,:,:]=autoencoder.predict(preds[i:i+1,:,:], batch_size=1)
    preds=preds*(X.max()-X.min())+X.min()

    print("[plotting...]")
    print("x: %s, preds: %s" % (X.shape, preds.shape))
    plt.plot(X[:,0,2], label='x')
    plt.plot(preds[:,0,2], label='predict')
    np.savetxt('ЙEМи.csv', preds.reshape(1200,3), delimiter = ',')
    plt.legend()
    plt.show()
    history.loss_plot('epoch')