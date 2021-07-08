# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:27:53 2021

@author: donte
"""
import sys
import matplotlib.pyplot as plt
import pickle
from Visualization import SVM_visualize_results
import torch
import torch.nn.functional as F
from sklearn import svm
def GetInput():
    print("choose data set:")
    print("(1) for MNIST")
    print("(2) for Fashion MNIST")
    choice= input()
    if(choice=='1'):
        dataset='mnist' # choose 'mnist' or 'fashionmnist'
    elif(choice=='2'):
        dataset='fashionmnist'
    else:
          sys.exit(0) 
    print("train or load model:")
    print("(1) for load model")
    print("(2) for train model")
    choice= input()
    if(choice=='1'):
        LoadModel=True
    elif(choice=='2'):
         LoadModel=False
    else:
         sys.exit(0) 
    print("choose amount of labeled:")
    print("(1) for 100")
    print("(2) for 600")
    print("(3) for 1000")
    print("(4) for 3000")
    choice= input()
    if  (choice=='1'):
         labeled_images=100
    elif(choice=='2'):
        labeled_images=600
    elif(choice=='3'):
        labeled_images=1000
    elif(choice=='4'):
        labeled_images=3000
    else:
         sys.exit(0)
    return dataset,LoadModel,labeled_images
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD
def train(vae,train_loader,device,optimizer,epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
   

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
def test(vae,test_loader,device):
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def precision(vae,test_loader,model):
    x=next(iter(test_loader))
    X=x[0]
    Y=x[1].numpy()
    recon_batch, mu, log_var = vae(X)
    x=vae.sampling(mu,log_var)
    X=x.detach().numpy()
    y=Y
    F=model.predict(X)
    prec=sum(y-F==0)/X.shape[0]*100
    print ("Model Precision",prec,'%' )
def hist(Y):
   
    plt.hist(Y, bins = 10)
    plt.show()
    
def SVMClassifier(vae,train_loader):
   x=next(iter(train_loader))
   X=x[0]
   Y=x[1].numpy()
   recon_batch, mu, log_var = vae(X)
   x=vae.sampling(mu,log_var)
   X=x.detach().numpy()
   y=Y
   # plt.figure()
   # hist(y)
   # sum(y==1)/y.shape[0]
   C = 1.0  # SVM regularization parameter
   model=svm.SVC(kernel='linear', gamma='auto', C=C)
   model =model.fit(X,y)   
   return model
def train_evaluate_save(labeled_images,vae,train_labeled_loader,test_loader,SVM_filename):
   print("train SVM model. labeled images:",labeled_images)
   SVM_model=SVMClassifier(vae,train_labeled_loader)#train SVM model 
   precision(vae,test_loader,SVM_model)# calculate precision
   if vae.fc31.out_features==2:
       SVM_visualize_results(vae,test_loader,SVM_model)
   pickle.dump(SVM_model, open(SVM_filename, 'wb'))
   print("SVM model saved")   
# SVM_model=SVMClassifier(vae,train_labeled_loader)#train SVM model 
# precision(vae,test_loader,SVM_model)