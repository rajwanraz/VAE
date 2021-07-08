import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.cm as cm
def visualize(vae,test_loader):
    x=next(iter(test_loader))
    X=x[0]
    index=0
    recon_batch, mu, log_var = vae(X)
    recon_batch=torch.reshape(recon_batch,[recon_batch.shape[0],28,28]).detach().numpy() 
    plt.imshow(recon_batch[index,:,:],cmap='gray')
    plt.figure()
    plt.imshow(X.detach().numpy()[index,0,:,:],cmap='gray')
def scatter_Z(vae,test_loader,label_data=True):
         x=next(iter(test_loader))
         X=x[0]
         Y=x[1]
         recon_batch, mu, log_var = vae(X)
         z=vae.sampling(mu,log_var)
         z=z.detach().numpy()
         if label_data==False:
            
             print ("mean:",z.mean(0))
             print ("std:",z.std(0))
             plt.scatter(z[:,0],z[:,1])

         else:
             ys = [i for i in range(10)]

             colors = cm.rainbow(np.linspace(0, 1, len(ys)))

             for  label, c in zip(ys, colors):
                 z_labeled=z[Y==label]
                 plt.scatter(z_labeled[:,0],z_labeled[:,1])
                 plt.xlabel('Sepal length')
                 plt.ylabel('Sepal width')
                 plt.title('latent representation of the test set')

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours( clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out


def SVM_visualize_results(vae,test_loader,model):  
    x=next(iter(test_loader))
    X=x[0]
    Y=x[1].numpy()
    recon_batch, mu, log_var = vae(X)
    x=vae.sampling(mu,log_var)
    X=x.detach().numpy()
    y=Y 
    title=  'SVC with linear kernel'
    # Set-up 2x2 grid for plotting.
    plt.figure()
  
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1) 
    plot_contours( model, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.title(title)
    
    plt.show()
    return model
def Generate_Z_samples(vae,device,z_dim):
    with torch.no_grad():
        image_size=28
        num_of_samples=64
        a=8
        z = torch.randn(num_of_samples,z_dim).to(device)
        sample = vae.decoder(z).to(device)
        samples=(sample.view(num_of_samples, 1, image_size, image_size))
        output=torch.zeros((a*image_size,a*image_size))
        counter=0
        for i in range(0,a):
            for j in range (0,a):
                output[i*image_size:(i+1)*image_size,j*image_size:(j+1)*image_size]=samples[counter,0,:,:]
                counter+=1
        plt.figure()
        plt.imshow(output.detach().numpy(),cmap='gray')




