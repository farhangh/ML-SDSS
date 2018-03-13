#!/usr/local/bin/python

# The code implements a MLP on the SDSS DR12 spectroscopic data to separate stars, galaxies and QSOs and computes the efficiency of the classification.

# F. Habibi, D. Mendes L'ete 2016 LaL

import numpy as np # array tools
import atpy as ap  # astronomical tables
from sklearn import linear_model as lm  # logistic regression
from matplotlib import pylab as plt # plot tools
import argparse # To read arguments from the console
import sys # try, exit, ...
print 
from sklearn.ensemble import RandomForestClassifier as rfc

#import astropy 
from astropy.io import fits
from astropy.table import Table 
#from astropy.coordinates import SkyCoord
from astropy import units as u


from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adadelta, RMSprop, Adagrad, Nadam
from sklearn.datasets import make_blobs
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.regularizers import l2, l1 #, c, dropout


import healpy as hp
from collections import Counter

import specphotoDefs8 as spd

#np.set_printoptions(threshold='nan')


#################### Main program #####################################
#def main():
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("-objc", help="Objects to be classified: qgs: Quasar-Galaxy-Star, gs: Galaxy-Star, qg: Quasar-Galaxy, qs: Quasar-Star", nargs='?')
    parser.add_argument('N', default=.5, help='Number of training object per type', nargs='?')
    parser.add_argument('f', default='test.dat', help='File name to write efficinecies vs. Ntrain', nargs='?')
    inputs = parser.parse_args()
    obj2classif = inputs.objc
    #nTrain = float(inputs.N) # Fraction of whole data considered as training set
    nTrain = int(inputs.N)
    file = inputs.f
    if not obj2classif : obj2classif = "qgs"

    print ''
    print ' ------------------------- specphotoClassif8.py ------------------------- '
    print '       Object classification using SDSS DR12 spectroscopy main sample '
    print '       Including classification efficiency and purity in Mollweide Map                   '
    print '                             KERAS Neural Network                               '
    print ' ---------------------------------------------------------------------------'
    print ''

    classistate = "None!"
    if obj2classif=='qgs': classistate = 'Quasar-Galaxy-Star'
    elif obj2classif=='qg': classistate = 'Quasar-Galaxy'
    elif obj2classif=='qs': classistate = 'Quasar-Star'
    elif obj2classif=='gs': classistate = 'Galaxy-Star'
    else:
        print 'ERROR: No specified objects to be classified! '
        print 'Exiting the peogram ...'
        sys.exit(0)
    print'Objects to be classified:', classistate
    print ''
    
    np.random.seed()

    # input data
    sample = 'dr12sgqSub5.fit' # zWarning + clean + type + nChild added
    print '1. Opening the SDSS main spec-photo sample ', sample, ' ...'
    #inp = fits.open(sample) # reading the fit file as an astropy table
    #ttbl = inp[1].data
    #tbl = Table.read(sample)
    ttbl = ap.Table(sample) # reading the fit file as an atpy table
    tbl = ttbl

    sNum = (ttbl.where((tbl['class']=='STAR'))).shape[0]
    gNum = (ttbl.where((tbl['class']=='GALAXY'))).shape[0]
    qNum = (ttbl.where((tbl['class']=='QSO'))).shape[0]
    print ' Initial data:'
    print '   #stars=', sNum, ', #galaxies=', gNum, ', #QSO=', qNum
    tot =sNum+gNum+qNum
    print '   #objects=',tot
    print '   sFrac=', round(sNum*1./tot,2), ', gFrac=', round(gNum*1./tot,2), ', qFrac=', round(qNum*1./tot,2)

# Filtering the low quality data
#    pp = .1
#    tbl = tbl.where( (tbl['objID']>0) & (tbl['magErr_u']<pp) & (tbl['magErr_g']<pp) & (tbl['magErr_r']<pp) & (tbl['magErr_i']<pp) & (tbl['magErr_z']<pp) )
#    tbl = tbl.where( (tbl.extinction_u<4) & (tbl['objID']>0) )

# Separating the three classes according to their spectral attribution
    print '2. Making separate tables for', classistate
    stbl = tbl.where((tbl['class']=='STAR'))
    gtbl = tbl.where((tbl['class']=='GALAXY'))
    qtbl = tbl.where((tbl['class']=='QSO'))#&((tbl['mag_g']>20)|(tbl['psfMag_z']/tbl['mag_z']<1.05)))

    sNum = stbl.shape[0] # Number of the stars
    gNum = gtbl.shape[0] #Number of the galaxies
    qNum = qtbl.shape[0] #Number of the QSOs
    print '   #stars=', sNum, ', #galaxies=', gNum, ', #QSO=', qNum
    tot =sNum+gNum+qNum
    print '   #objects=',tot
    print '   sFrac=', round(sNum*1./tot,2), ', gFrac=', round(gNum*1./tot,2), ', qFrac=', round(qNum*1./tot,2)



    mtbl = stbl
    mtbl.append(gtbl)
    mtbl.append(qtbl)
    maintbl = spd.GetMainTbl(mtbl,obj2classif) # watch out the condition for the point-like sources    
    indices = np.random.randint(low=0, high=maintbl.shape[0], size=maintbl.shape[0])
    #maintbl = maintbl[indices] # Shuffling the maintbl

    # Trainig sample selection
    print '3. We select', nTrain, 'of',classistate,'to make a weighted trainig sample ...'
    indices = np.random.randint(low=0, high=maintbl.shape[0], size=nTrain)
    trainSample = maintbl[indices]

    # Validation sample selection
    validationSample = np.delete(maintbl,indices)
    valIndices = np.random.randint(low=0, high=validationSample.shape[0], size=nTrain)
    validationSample = validationSample[valIndices]
    

    print '4. Making the feature matrix (including bias factors) from the traning sample ...'
    # Constructing the feature matrix for the training sample
    X = spd.getFeatures(trainSample)
    X_valid = spd.getFeatures(validationSample)
    #meanX = X.mean(axis=0) # mean per each column
    #spanX = abs(X).max(axis=0) #X.max(axis=0)-X.min(axis=0)
    #X -= meanX
    #X /= spanX

    print '5. Making the model vector from the traning sample ...'
    # Labeling the stars and galaxies
    Y = spd.getModel(trainSample,obj2classif)
    Y_valid = spd.getModel(validationSample,obj2classif)

    indices = np.random.randint(low=0, high=validationSample.shape[0], size=10000)
    validSkeras = validationSample #[indices]
    xV = spd.getFeatures(validSkeras)
    xV = xV[:,1:]
    yV = spd.getModel(validSkeras,obj2classif)


    print '6. Classification with NeuralNetworks...'
	
    print 'building feature matrix X...'
	# convert output classes (integers) to one hot vectors.
    # this is necessary for keras. if you have 3 classes,
    # for instance, keras expects y to be a matrix
    # of nb_examples rows and 3 columns. 
    # for each row, the column i (starting from zero) 
    # is 1 if the class is i, otherwise it is 0
    # for instance, the class 0 would be converted to
    # [1 0 0], the class 1 to [0 1 0] and the class 2
    # to [0 0 1]
    y = to_categorical(Y, nb_classes=3)
    y_valid = to_categorical(yV, nb_classes=3)
    print 'building classes matrix Y...'
    X = X[:,1:]
    X_valid = X_valid[:,1:]

    print'construct the architecture of the neural network...'
    #model = spd.build_model_keras( onu )            # construct the architecture of the neural net
    
    lambda_l2 = 1.e-5	# parametre de regularisation
    print '     Regularisation factor:', lambda_l2
    model = Sequential()

    nfeat = 10
    print 'Number of features=',nfeat
    model.add(Dense(50, activation='tanh', W_regularizer=l2(lambda_l2), input_dim=nfeat))	# 50 units in first layer, 7 is nb of input variables
    model.add(Dense(20, activation='tanh', W_regularizer=l2(lambda_l2))) 				# 20 units in second layer
    #model.add(Dropout(0.5)) 															# uncomment to apply dropout after the second layer

    model.add(Dense(3, activation='softmax')) 										# 3 units in output layer (nb of classes)
	
    #model.get_config()
    print 'save the model into "model.pkl"...'
    model_filename = 'model.pkl'                     # save the model in this file
    
    #Before training a model, you need to configure the learning process, which is done via the compile method. It receives three arguments:
    # a loss function. This is the objective that the model will try to minimize. It can be the string identifier of an existing loss function (such as categorical_crossentropy or mse), or it can be an objective function. See: objectives.
 
    # an optimizer. This could be the string identifier of an existing optimizer (such as rmsprop or adagrad), or an instance of the Optimizer class. See: http://keras.io/optimizers/
 
    # a list of metrics. For any classification problem you will want to set this to metrics=['accuracy']. A metric could be the string identifier of an existing metric (only accuracy is supported at this point), or a custom metric function.
    
    onu = 3
    nb_epoch = 900
    batch_size = 128
    initial_lr = 2.e-3
    print 'compiling the model...'
    model.compile(loss='categorical_crossentropy', 
                  #optimizer=Adadelta(lr=initial_lr),
                  #optimizer=RMSprop(lr=initial_lr),
                  optimizer=Nadam(lr=initial_lr,schedule_decay=0.004),
                  metrics=['mae', 'acc']) #http://keras.io/optimizers/
#metrics=['mae', 'acc']


# This is the function  for adjust the learning rate during training
    def schedule(epoch):
        old_lr = model.optimizer.lr.get_value()
        # linear decay
        #decay = (.1-.01)/300. #1e-2
        #new_lr = initial_lr -epoch*decay #/ (1. + decay * epoch)
        new_lr = float(old_lr)  #float(new_lr)
        return new_lr
                
                    
    # train the model
    print 'setting up learning parameter...'
    callbacks = [                                            # A callback is a set of functions to be applied at given stages of the training procedure.
        EarlyStopping(monitor = 'val_acc',                     # quantity to be monitored.
                      patience = 50,                         # number of epochs with no improvement after which training will be stopped.
                      verbose = 1,
                      mode = 'auto'),                        # one of {auto, min, max}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored has stopped increasing.
        ModelCheckpoint(model_filename,                     # Save the model after every epoch.
                        monitor = 'val_acc',                 # quantity to monitor.
                        save_best_only = True,                 # if save_best_only=True, the latest best model according to the validation loss will not be overwritten.
                        mode = 'auto'),                        # one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minization of the monitored. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity./Users/farhang/Library/Containers/com.apple.mail/Data/Library/Mail Downloads/70564FC5-36A5-4465-9A5F-FC79FA45EAA6/20170104_154652.pdf
        LearningRateScheduler(schedule)
                 # This is the function for adjusting the learning rate during training, GoTo specphotoDefs8
                 ]
    print 'start learning...'

    history = model.fit(X, y,                                 # training process, with here the training sample
                        batch_size = batch_size,            # the batch size of the training sample 
                        nb_epoch = nb_epoch,                # number of epoch of the training
                        callbacks = callbacks,                # Callbaks during the training session
                        verbose = 1,                        # Info to show
                        validation_data=(xV,y_valid)
                        )
#    print 'load model with best validation accuracy...'
#    model.load_weights(model_filename)                     # load the model in the epoch which gave the best validation accuracy
        
#    score = model.evaluate(X, y, verbose=1)         # evaluate on test data
    
    #    print 'Test score:', score[0]
    #    print 'Test accuracy:', score[1]
    
 

    predY = model.predict_classes(X)
    print '   Efficiency of the training sample (NN):'
    #    spd.GetObjEfficiency(predY,testY,obj2classif)
    spd.GetObjEfficiency(predY,Y,obj2classif)
    pTrain = spd.GetObjEfficiency(predY,Y,obj2classif)
    NT = spd.GetObjBlending(predY, Y, obj2classif) # [sg,sq,gs,gq,qs,qg]
    ind = np.where(Y==0)
    sNumTrain = Y[ind].shape[0]
    ind = np.where(Y==1)
    gNumTrain = Y[ind].shape[0]
    ind = np.where(Y==2)
    qNumTrain = Y[ind].shape[0]
    
    PureTrain = spd.ComputePurity(NT, pTrain, sNumTrain, gNumTrain, qNumTrain, obj2classif) #[s,g,q]
    #PureTrain = spd.ComputePurity(NT, pTrain, sNum, gNum, qNum, obj2classif) #[s,g,q]
    if(obj2classif=='qgs'):
        print'   Star purity:', PureTrain[0]*100,'%'
        print'   Galaxy purity:', PureTrain[1]*100,'%'
        print'   QSO purity:', PureTrain[2]*100,'%'

    outY = model.predict_classes(X_valid)
    controlY = spd.getModel(validationSample,obj2classif)
    print'   Efficiency of the validation sample:'
    pValid = spd.GetObjEfficiency(outY,controlY,obj2classif) # [stars, galaxies, QSOs, total]

    N = spd.GetObjBlending(outY, controlY, obj2classif) # [sg,sq,gs,gq,qs,qg]
    ind = np.where(controlY==0)
    sNumValid = controlY[ind].shape[0]
    ind = np.where(controlY==1)
    gNumValid = controlY[ind].shape[0]
    ind = np.where(controlY==2)
    qNumValid = controlY[ind].shape[0]
    
    PureValid = spd.ComputePurity(N, pValid, sNumValid, gNumValid, qNumValid, obj2classif) #[s,g,q]
    #Pure = spd.ComputePurity(N, pMain, sNum, gNum, qNum, obj2classif) #[s,g,q]
    if(obj2classif=='qgs'):
        print'   Star purity:', round(PureValid[0],3)*100,'%'
        print'   Galaxy purity:', round(PureValid[1],3)*100,'%'
        print'   QSO purity:', round(PureValid[2],3)*100,'%'



    XX = spd.getFeatures(maintbl)
    XX = XX[:,1:]
    outY = model.predict_classes(XX)
    controlY = spd.getModel(maintbl,obj2classif)

    print
    print '7. Converting equatorial coordinates to Galactic coordinates ... '

    ra = maintbl['ra']
    dec = maintbl['dec']
    c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    gcoord = c_icrs.galactic
    l = gcoord.l.value
    b = gcoord.b.value


    objInfo = np.zeros([l.shape[0],12])
    objInfo[:,0] = l # phi
    objInfo[:,1] = b # theta
    objInfo[:,2] = controlY
    objInfo[:,3] = outY
    objInfo[:,4] = maintbl['mag_u']-maintbl['extinction_u']
    objInfo[:,5] = maintbl['mag_g']-maintbl['extinction_g']
    objInfo[:,6] = maintbl['mag_r']-maintbl['extinction_r']
    objInfo[:,7] = maintbl['mag_i']-maintbl['extinction_i']
    objInfo[:,8] = maintbl['mag_z']-maintbl['extinction_z']
    objInfo[:,9] = maintbl['mag_z']/maintbl['psfMag_z'] #maintbl['deVRad_z']
    objInfo[:,10] = maintbl['z']
    
    objInfo[:,0] *= np.pi/180.
    objInfo[:,1] = (90.-objInfo[:,1])*np.pi/180.

    #np.savetxt('objInfo.dat',objInfo)


    col1 = fits.Column(name='l', format='E', array=l*np.pi/180.)
    col2 = fits.Column(name='b', format='E', array=(90.-b)*np.pi/180.)
    col3 = fits.Column(name='conty', format='E', array=controlY)
    col4 = fits.Column(name='outy', format='E', array=outY)
    col5 = fits.Column(name='u', format='E', array=maintbl['mag_u']-maintbl['extinction_u'])
    col6 = fits.Column(name='g', format='E', array=maintbl['mag_g']-maintbl['extinction_g'])
    col7 = fits.Column(name='r', format='E', array=maintbl['mag_r']-maintbl['extinction_r'])
    col8 = fits.Column(name='i', format='E', array=maintbl['mag_i']-maintbl['extinction_i'])
    col9 = fits.Column(name='z', format='E', array=maintbl['mag_z']-maintbl['extinction_z'])
    col10 = fits.Column(name='redshift', format='E', array=maintbl['z'])
    col11 = fits.Column(name='rad', format='E', array=maintbl['mag_z']/maintbl['psfMag_z'])
    col12 = fits.Column(name='subClass', format='20A', array=maintbl['subClass'])
    col13 = fits.Column(name='subClassFlag', format='I', array=maintbl['subClassFlag'])
    col14 = fits.Column(name='subClassFlag2', format='I', array=maintbl['subClassFlag2'])
    col15 = fits.Column(name='ra', format='E', array=ra)
    col16 = fits.Column(name='dec', format='E', array=dec)
    col17 = fits.Column(name='delu', format='E', array=maintbl['magErr_u'])
    col18 = fits.Column(name='delg', format='E', array=maintbl['magErr_g'])
    col19 = fits.Column(name='delr', format='E', array=maintbl['magErr_r'])
    col20 = fits.Column(name='deli', format='E', array=maintbl['magErr_i'])
    col21 = fits.Column(name='delz', format='E', array=maintbl['magErr_z'])
    col22 = fits.Column(name='extu', format='E', array=maintbl['extinction_u'])
    col23 = fits.Column(name='extg', format='E', array=maintbl['extinction_g'])
    col24 = fits.Column(name='extr', format='E', array=maintbl['extinction_r'])
    col25 = fits.Column(name='exti', format='E', array=maintbl['extinction_i'])
    col26 = fits.Column(name='extz', format='E', array=maintbl['extinction_z'])
    col27 = fits.Column(name='id', format='K', array=maintbl['objID'])
    col28 = fits.Column(name='nchild', format='E', array=maintbl['nChild'])
    col29 = fits.Column(name='clean', format='E', array=maintbl['clean'])
    col30 = fits.Column(name='type', format='E', array=maintbl['type'])


    cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    print ' Writting the classification results in objInfo.fits ...'
    #tbhdu.writeto('objInfo.fits',clobber='True')
    #tbhdu.writeto('objInfoS.fits',clobber='True')



    print '8. Constructing Galaxies purity in a healpix map ...'
    nside = 64
    print '   nside=',nside
    print '   constructing map of total number of objects ... '
    NtotMap = spd.ConstructMap(nside,objInfo)
                        
    #hp.mollview(NtotMap, title="Test")
    print '   constructing separate map of number of stars, galaxies and QSOs ... '
    # objects real classes
    NsMap = spd.ComputeNaMap(objInfo,nside,0)
    NgMap = spd.ComputeNaMap(objInfo,nside,1)
    NqMap = spd.ComputeNaMap(objInfo,nside,2)

    print '   constructing maps for objects fraction ... '
    NsFracMap = spd.ComputeNaFraction(NtotMap,NsMap)
    NgFracMap = spd.ComputeNaFraction(NtotMap,NgMap)
    NqFracMap = spd.ComputeNaFraction(NtotMap,NqMap)



    # stars (mis)classified to s,g,q
    NssMap = spd.ComputeNabMap(objInfo,nside,0,0)
    NsgMap = spd.ComputeNabMap(objInfo,nside,0,1)
    NsqMap = spd.ComputeNabMap(objInfo,nside,0,2)

    # galaxies (mis)classified to s,g,q
    NggMap = spd.ComputeNabMap(objInfo,nside,1,1)
    NgsMap = spd.ComputeNabMap(objInfo,nside,1,0)
    NgqMap = spd.ComputeNabMap(objInfo,nside,1,2)

    # QSOs (mis)classified to s,g,q
    NqqMap = spd.ComputeNabMap(objInfo,nside,2,2)
    NqsMap = spd.ComputeNabMap(objInfo,nside,2,0)
    NqgMap = spd.ComputeNabMap(objInfo,nside,2,1)



    print '   constructing map of number of misclasified objects ... '

    # stars and QSOs classified as galaxies
    NqsgMap = spd.ComputeNsqgMap(NsgMap,NqgMap)
    # stars and galaxis classified as QSOs
    NsgqMap = spd.ComputeNsqgMap(NsqMap,NgqMap)
    # galaxies and QSOs classified as stars
    NgqsMap = spd.ComputeNsqgMap(NgsMap,NqsMap)



    print '   computing the star, galaxy and QSO purity maps ...'
    sPureMap = spd.ComputePureMap(NssMap,NgqsMap)
    gPureMap = spd.ComputePureMap(NggMap,NqsgMap)
    qPureMap = spd.ComputePureMap(NqqMap,NsgqMap)



    print '   computing the star, galaxy and QSO  efficiency map ...'
    
    sEfficMap = spd.ComputeEfficMap(NsMap,NssMap)
    gEfficMap = spd.ComputeEfficMap(NgMap,NggMap)
    qEfficMap = spd.ComputeEfficMap(NqMap,NqqMap)


    hp.write_map("NtotMapNN.fits", NtotMap)
    hp.write_map("NggMapNN.fits", NggMap)

    hp.write_map("NsMapNN.fits", NsMap)
    hp.write_map("NgMapNN.fits", NgMap)
    hp.write_map("NqMapNN.fits", NqMap)

    hp.write_map("NsFracMapNN.fits", NsFracMap)
    hp.write_map("NgFracMapNN.fits", NgFracMap)
    hp.write_map("NqFracMapNN.fits", NqFracMap)

    hp.write_map("NsgMapNN.fits", NsgMap)
    hp.write_map("NqgMapNN.fits", NqgMap)
    hp.write_map("NsqgMapNN.fits", NqsgMap)
    hp.write_map("gPureMapNN.fits", gPureMap)
    hp.write_map("gEfficMapNN.fits", gEfficMap)
    hp.write_map("sPureMapNN.fits", sPureMap)
    hp.write_map("sEfficMapNN.fits", sEfficMap)
    hp.write_map("qPureMapNN.fits", qPureMap)
    hp.write_map("qEfficMapNN.fits", qEfficMap)


    print '9. Saving map value of galaxy purity, number of stars, galaxies and QSOs in gPureNsNgNq.dat ... '
    [theta, phi] = hp.pix2ang(nside,range(NgMap.shape[0]))
    theta *= 180./np.pi # b
    phi *= 180./np.pi # l
    nt = np.ndarray([NgMap.shape[0],11])
    nt[:,0] = sPureMap[:]
    nt[:,1] = gPureMap[:]
    nt[:,2] = qPureMap[:]
    nt[:,3] = NsMap[:]
    nt[:,4] = NgMap[:]
    nt[:,5] = NqMap[:]
    nt[:,6] = sEfficMap[:]
    nt[:,7] = gEfficMap[:]
    nt[:,8] = qEfficMap[:]
    nt[:,9] = phi[:] # l
    nt[:,10] = 90.-theta[:] # b
    np.savetxt('effpureNsNgNq.dat',nt)




except KeyboardInterrupt:
    print ''
    print "Shutdown requested. Exiting ... "



print ''
print '--------------- End ------------------ '
