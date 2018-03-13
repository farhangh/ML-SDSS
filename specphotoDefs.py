import numpy as np # array tools
import atpy as ap  # astronomical tables

import healpy as hp
from collections import Counter



######################## Defined functions ###################

def getFeatures(mytbl):
    # we construct the feature matrix using the colours and their quadratic mutiplications
    n = 10   # number of features (colors)
    m = mytbl.shape[0]
    wX = np.ndarray([m,n+1])
    
    wX[:,0] = 1.
    '''
    wX[:,1] = mytbl['mag_u']-mytbl['mag_g'] - (mytbl['extinction_u']-mytbl['extinction_g'])
    wX[:,2] = mytbl['mag_g']-mytbl['mag_r'] - (mytbl['extinction_g']-mytbl['extinction_r'])
    wX[:,3] = mytbl['mag_r']-mytbl['mag_i'] - (mytbl['extinction_r']-mytbl['extinction_i'])
    wX[:,4] = mytbl['mag_i']-mytbl['mag_z'] - (mytbl['extinction_i']-mytbl['extinction_z'])

    wX[:,5] = mytbl['psfMag_u']- mytbl['psfMag_g'] - (mytbl['extinction_u']-mytbl['extinction_g'])
    wX[:,6] = mytbl['psfMag_g']- mytbl['psfMag_r'] - (mytbl['extinction_g']-mytbl['extinction_r'])
    wX[:,7] = mytbl['psfMag_r']- mytbl['psfMag_i'] - (mytbl['extinction_r']-mytbl['extinction_i'])
    wX[:,8] = mytbl['psfMag_i']- mytbl['psfMag_z'] - (mytbl['extinction_i']-mytbl['extinction_z'])

    wX[:,9] = mytbl['modelMag_u']-mytbl['modelMag_g'] - (mytbl['extinction_u']-mytbl['extinction_g'])
    wX[:,10] = mytbl['modelMag_g']- mytbl['modelMag_r'] - (mytbl['extinction_g']-mytbl['extinction_r'])
    wX[:,11] = mytbl['modelMag_r']- mytbl['modelMag_i'] - (mytbl['extinction_r']-mytbl['extinction_i'])
    wX[:,12] = mytbl['modelMag_i']- mytbl['modelMag_z'] - (mytbl['extinction_i']-mytbl['extinction_z'])
    
    
    wX[:,13] = (mytbl['mag_u']-mytbl['extinction_u'])/(mytbl['psfMag_u']-mytbl['extinction_u'])
    wX[:,14] = (mytbl['mag_g']-mytbl['extinction_g'])/(mytbl['psfMag_g']-mytbl['extinction_g'])
    wX[:,15] = (mytbl['mag_r']-mytbl['extinction_r'])/(mytbl['psfMag_r']-mytbl['extinction_r'])
    wX[:,16] = (mytbl['mag_i']-mytbl['extinction_i'])/(mytbl['psfMag_i']-mytbl['extinction_i'])
    wX[:,17] = (mytbl['mag_z']-mytbl['extinction_z'])/(mytbl['psfMag_z']-mytbl['extinction_z'])

    wX[:,18] = mytbl['psfMag_u']- mytbl['mag_u']
    wX[:,19] = mytbl['psfMag_g']- mytbl['mag_g']
    wX[:,20] = mytbl['psfMag_r']- mytbl['mag_r']
    wX[:,21] = mytbl['psfMag_i']- mytbl['mag_i']
    wX[:,22] = mytbl['psfMag_z']- mytbl['mag_z']
    

    wX[:,23] = mytbl['psfMag_u'] - mytbl['extinction_u']
    wX[:,24] = mytbl['mag_u'] - mytbl['extinction_u']
    '''

    '''
    wX[:,1] = mytbl['mag_u'] - mytbl['extinction_u']
    wX[:,2] = mytbl['mag_g'] - mytbl['extinction_g']
    wX[:,3] = mytbl['mag_r'] - mytbl['extinction_r']
    wX[:,4] = mytbl['mag_i'] - mytbl['extinction_i']
    wX[:,5] = mytbl['mag_z'] - mytbl['extinction_z']

    wX[:,6] = mytbl['psfMag_u'] - mytbl['extinction_u']
    wX[:,7] = mytbl['psfMag_g'] - mytbl['extinction_g']
    wX[:,8] = mytbl['psfMag_r'] - mytbl['extinction_r']
    wX[:,9] = mytbl['psfMag_i'] - mytbl['extinction_i']
    wX[:,10] = mytbl['psfMag_z'] - mytbl['extinction_z']
    
    wX[:,11] = mytbl['modelMag_u'] - mytbl['extinction_u']
    wX[:,12] = mytbl['modelMag_g'] - mytbl['extinction_g']
    wX[:,13] = mytbl['modelMag_r'] - mytbl['extinction_r']
    wX[:,14] = mytbl['modelMag_i'] - mytbl['extinction_i']
    wX[:,15] = mytbl['modelMag_z'] - mytbl['extinction_z']
    '''
    
    
    wX[:,1] = (mytbl['mag_u']-mytbl['extinction_u'])/(mytbl['psfMag_u']-mytbl['extinction_u'])
    wX[:,2] = (mytbl['mag_g']-mytbl['extinction_g'])/(mytbl['psfMag_g']-mytbl['extinction_g'])
    wX[:,3] = (mytbl['mag_r']-mytbl['extinction_r'])/(mytbl['psfMag_r']-mytbl['extinction_r'])
    wX[:,4] = (mytbl['mag_i']-mytbl['extinction_i'])/(mytbl['psfMag_i']-mytbl['extinction_i'])
    wX[:,5] = (mytbl['mag_z']-mytbl['extinction_z'])/(mytbl['psfMag_z']-mytbl['extinction_z'])

    wX[:,6] = mytbl['psfMag_u']- mytbl['mag_u']
    wX[:,7] = mytbl['psfMag_g']- mytbl['mag_g']
    wX[:,8] = mytbl['psfMag_r']- mytbl['mag_r']
    wX[:,9] = mytbl['psfMag_i']- mytbl['mag_i']
    wX[:,10] = mytbl['psfMag_z']- mytbl['mag_z']
    


    
    return wX


def getModel(mytbl,obj2classif):
    m = mytbl.shape[0]
    wY = np.ndarray(m)
    # Stars are labeld as '0', galaxies as '1' and QSOs as '3'
    if obj2classif=='qgs':
        starIndex = np.where(mytbl['class']=='STAR')
        galIndex =  np.where(mytbl['class']=='GALAXY')
        qsoIndex =  np.where(mytbl['class']=='QSO')
        wY[starIndex] = 0
        wY[galIndex] = 1
        wY[qsoIndex] = 2
    elif obj2classif=='gs':
        starIndex = np.where(mytbl['class']=='STAR')
        galIndex =  np.where(mytbl['class']=='GALAXY')
        wY[starIndex] = 0
        wY[galIndex] = 1
    elif obj2classif=='qg':
        qsoIndex =  np.where(mytbl['class']=='QSO')
        galIndex =  np.where(mytbl['class']=='GALAXY')
        wY[qsoIndex] = 2
        wY[galIndex] = 1
    elif obj2classif=='qs':
        starIndex = np.where(mytbl['class']=='STAR')
        qsoIndex =  np.where(mytbl['class']=='QSO')
        wY[starIndex] = 0
        wY[qsoIndex] = 2
    else:
        print( 'GetModel ERROR: Objects types to be classified not specified.')
    
    
    return wY



def GetTrainSample(nT, stbl,Sindices, gtbl,Gindices, qtbl, Qindices, obj2classif):
    if obj2classif=='qgs':
        trainSample = stbl.rows(Sindices) # making a random star sample with nTrain stars
        trainSample.append(gtbl.rows(Gindices)) # appending the random galaxy sample to the random star sample
        trainSample.append(qtbl.rows(Qindices)) # appending the random quasar sample

    elif obj2classif=='gs':
        trainSample = stbl.rows(Sindices) # making a random star sample with nTrain stars
        trainSample.append(gtbl.rows(Gindices)) # appending the random galaxy sample to the random star sample

    elif obj2classif=='qg':
        trainSample = qtbl.rows(Qindices) # making a random star sample with nTrain stars
        trainSample.append(gtbl.rows(Gindices)) # appending the random galaxy sample to the random star sample

    elif obj2classif=='qs':
        trainSample = qtbl.rows(Qindices) # making a random star sample with nTrain stars
        trainSample.append(stbl.rows(Sindices)) # appending the random galaxy sample to the random star sample

    else:
        print 'GetTrainSample ERROR: Objects types to be classified not specified.'
    
    return trainSample




def GetWeightedTrainSample(nT, stbl, gtbl, qtbl, obj2classif):

    sNum = stbl.shape[0]
    gNum = gtbl.shape[0]
    qNum = qtbl.shape[0]
    
    if obj2classif=='qgs':

        sFrac = sNum*1./(gNum+sNum+qNum)
        sNrand = int(nT*sFrac)
        sIndices = np.random.rand(sNrand)*sNum
        sIndices = sIndices.astype(np.int64)

        gFrac = gNum*1./(gNum+sNum+qNum)
        gNrand = int(nT*gFrac)
        gIndices = np.random.rand(nT)*gNum
        gIndices = gIndices.astype(np.int64)

        qFrac = qNum*1./(gNum+sNum+qNum)
        qNrand = int(nT*qFrac)
        qIndices = np.random.rand(qNrand)*qNum
        qIndices = qIndices.astype(np.int64)

        trainSample = stbl.rows(sIndices) # making a random star sample with nTrain stars
        trainSample.append(gtbl.rows(gIndices)) # appending the random galaxy sample to the random star sample
        trainSample.append(qtbl.rows(qIndices)) # appending the random quasar sample
    
    elif obj2classif=='gs':
        sFrac = sNum*1./(gNum+sNum)
        sNrand = int(nT*sFrac)
        sIndices = np.random.rand(sNrand)*sNum
        sIndices = sIndices.astype(np.int64)
        
        gFrac = gNum*1./(gNum+sNum)
        gNrand = int(nT*gFrac)
        gIndices = np.random.rand(nT)*gNum
        gIndices = gIndices.astype(np.int64)
        
        trainSample = stbl.rows(sIndices) # making a random star sample with nTrain stars
        trainSample.append(gtbl.rows(gIndices)) # appending the random galaxy sample to the random star sample
    
    elif obj2classif=='qg':
        gFrac = gNum*1./(gNum+qNum)
        gNrand = int(nT*gFrac)
        gIndices = np.random.rand(nT)*gNum
        gIndices = gIndices.astype(np.int64)

        qFrac = qNum*1./(gNum+qNum)
        qNrand = int(nT*qFrac)
        qIndices = np.random.rand(qNrand)*qNum
        qIndices = qIndices.astype(np.int64)

        trainSample = qtbl.rows(qIndices) # making a random star sample with nTrain stars
        trainSample.append(gtbl.rows(gIndices)) # appending the random galaxy sample to the random star sample
    
    elif obj2classif=='qs':
        qFrac = qNum*1./(sNum+qNum)
        qNrand = int(nT*qFrac)
        qIndices = np.random.rand(qNrand)*qNum
        qIndices = qIndices.astype(np.int64)
        
        sFrac = sNum*1./(sNum+qNum)
        sNrand = int(nT*sFrac)
        sIndices = np.random.rand(sNrand)*sNum
        sIndices = sIndices.astype(np.int64)

        trainSample = qtbl.rows(qIndices) # making a random star sample with nTrain stars
        trainSample.append(stbl.rows(sIndices)) # appending the random galaxy sample to the random star sample
    
    else:
        print 'GetTrainSample ERROR: Objects types to be classified not specified.'
    
    return trainSample





def ComputeObjEfficiency(myY,controlY,objType):
    if objType == 3: dY = controlY-myY
    else:
        Index = np.where(controlY==objType)
        dY = controlY[Index]-myY[Index]
    dY0index = np.nonzero(dY)
    if (dY.shape[0]==0): p = 0.
    else: p = 1. - float(dY0index[0].shape[0])/dY.shape[0]
    return p



def GetObjEfficiency(newsY,Y,obj2classif):
    pp = []
    # Efficiency of the star classification
    if not (obj2classif=="qg"):
        p = ComputeObjEfficiency(newsY,Y,0)
        pp.append(round(p,3))
        print '   Stars:',round(p*100.,1),'%'

    # Efficiency of the galaxy classification
    if not (obj2classif=="qs"):
        p = ComputeObjEfficiency(newsY,Y,1)
        pp.append(round(p,3))
        print '   Galaxies:',round(p*100.,1),'%'

    # Efficiency of the quasar classification
    #if (obj2classif=="qs" | obj2classif=="qg" | obj2classif=="qgs" ):
    if not (obj2classif=="gs"):
        p = ComputeObjEfficiency(newsY,Y,2)
        pp.append(round(p,3))
        print '   QSOs:', round(p*100.,1),'%'

    p = ComputeObjEfficiency(newsY,Y,3)
    pp.append(round(p,3))
    print '   Total:',round(p*100.,1),'%'

    return pp




def ComputeBlending(newsY, controlY, objType, ab, ac, f='true'):
    N = []
    Index = np.where(controlY==objType)
    Yabc = newsY[Index]
    abInd = np.where(Yabc == ab)
    Nab = abInd[0].shape[0]
    N.append(Nab)
 
    if(f):
        acInd = np.where(Yabc == ac)
        Nac = acInd[0].shape[0]
        N.append(Nac)
    
    return N


def GetObjBlending(newsY,controlY,obj2classif):
    N = []
    
    if (obj2classif=="qgs"):
        # Purity of the star classification
        Nsgq = ComputeBlending(newsY, controlY, 0, 1, 2)
        N.append(Nsgq[0]) # s2g
        N.append(Nsgq[1]) # s2q
        # purity of the galaxy classification
        Ngsq = ComputeBlending(newsY, controlY, 1, 0, 2)
        N.append(Ngsq[0]) #g2s
        N.append(Ngsq[1]) #g2q
        # Purity of the quasar classification
        Nqsg = ComputeBlending(newsY, controlY, 2, 0, 1)
        N.append(Nqsg[0]) #q2s
        N.append(Nqsg[1]) #q2g
    
    if (obj2classif=="qs"):
        # Purity of the star classification
        Nsq = ComputeBlending(newsY, controlY, 0, 2, 'nan', 'false')
        N.append(Nsq[0]) # s2q
        Nqs = ComputeBlending(newsY, controlY, 2, 0, 'nan', 'false')
        N.append(Nqs[0]) # q2s
    

    return N


def ComputePurity(N, pMain, sNum, gNum, qNum, obj2classif):
    pure = []
    
    if(obj2classif=="qgs"):
        Nsg = float(N[0])
        Nsq = float(N[1])
        Ngs = float(N[2])
        Ngq = float(N[3])
        Nqs = float(N[4])
        Nqg = float(N[5])
    
        Nss = pMain[0]*sNum
        if(abs(Nss+Ngs+Nqs)<1.e-5): sPure = 0.
        else: sPure = round(Nss/(Nss+Ngs+Nqs),3)
        pure.append(sPure)
    
        Ngg = pMain[1]*gNum
        if(abs(Ngg+Nsg+Nqg)<1.e-5): gPure = 0.
        else: gPure = round(Ngg/(Ngg+Nsg+Nqg),3)
        pure.append(gPure)

        Nqq = pMain[2]*qNum
        if(abs(Nqq+Nsq+Ngq)<1.e-5): qPure = 0.
        else: qPure = round(Nqq/(Nqq+Nsq+Ngq),3)
        pure.append(qPure)

    if(obj2classif=="qs"):
        Nsq = float(N[0])
        Nqs = float(N[1])

        Nss = pMain[0]*sNum
        if(abs(Nss+Nqs)<1.e-5): sPure = 0.
        else: sPure = round(Nss/(Nss+Nqs),3)
        pure.append(sPure)

        Nqq = pMain[2]*qNum
        if(abs(Nqq+Nsq)<1.e-5): qPure = 0.
        else: qPure = round(Nqq/(Nqq+Nsq),3)
        pure.append(qPure)



    return pure



def ConstructMap(nside,objInfo):
    npix = hp.nside2npix(nside)
    Map = np.zeros(npix)
    Map -= 1. # non observed parts
    
    # keeping the pixel number of each object in a list
    pixNumber = (hp.ang2pix(nside, objInfo[:,1], objInfo[:,0])).tolist()
    
    # counting the number of repeated pixel numbers in the list
    citer = dict(Counter(pixNumber))
    n = citer.values() # number of repeatation of a pixel number
    pixInd = citer.keys() # pix nubmer on the map
    Map[pixInd] = n

    return Map


def ComputeNsqgMap(NsgMap,NqgMap):
    NsqgMap = np.zeros([NsgMap.shape[0]])
    NsqgMap -= 1.
    ind = np.where( (NsgMap>-1.)&(NqgMap<1.) )
    NsqgMap[ind] = NsgMap[ind]
    ind = np.where( (NsgMap<1.)&(NqgMap>-1.) )
    NsqgMap[ind] = NqgMap[ind]
    ind = np.where( (NsgMap>-1.)&(NqgMap>-1.) )
    NsqgMap[ind] = NsgMap[ind]+NqgMap[ind]

    return NsqgMap



def CopyMap(map):
    mmap = map * 1.
    return mmap



def ExtRowsTfW(Trows,Wnum): #row indices of training set, number of rows for the whole sample
    print( 'DEB: Wnum=',Wnum)
    Wrows = range(Wnum) # list of integer range from 0 to Wnum-1
    Wrows = np.asarray(Wrows) # list to array
    print( 'DEB: Wnum=',Wnum,', Trows.shape=',Trows.shape[0])
    print( 'DEB: ',Wrows)
    
    Erows = [] # an empty list
    print( 'DEB: ', Erows)
    for i in range(Wnum):
        print( 'DEB: i=',i)
        control = 0
        for j in range(Trows.shape[0]):
            if Wrows[i]==Trows[j]:
                control = 1
        if control == 0:
            Erows.append(Wrows[i])
    Erows = np.asarray(Erows)
    return Erows



def GetMainTbl(tbl,obj2classif):
    if obj2classif=="qgs":
        IndexMaintbl = ((tbl['class']=='STAR') | (tbl['class']=='GALAXY') | (tbl['class']=='QSO'))
        maintbl = tbl[IndexMaintbl]

    elif obj2classif=="gs":
        IndexMaintbl = ((tbl['class']=='STAR') | (tbl['class']=='GALAXY') )
        maintbl = tbl[IndexMaintbl]

    elif obj2classif=="qg":
        IndexMaintbl = ((tbl['class']=='GALAXY') | (tbl['class']=='QSO') )
        maintbl = tbl[IndexMaintbl]

    elif obj2classif=="qs":
        IndexMaintbl = ((tbl['class']=='STAR') | (tbl['class']=='QSO') )
        maintbl = tbl[IndexMaintbl]

    else:
        print( 'GetMainTbl ERROR: Objects types to be classified not specified.')
    return maintbl


def MakeFitsTable(inpTbl,Yclassif,Ycontrol,objType=0,bgFlag="bad",fitsFileName="badStars.fits"):
    index=np.where(Ycontrol==objType)
    #index = (Ycontrol == objType)
    gbIndex = 0
    if bgFlag=="bad" :
        gbIndex=np.nonzero(Yclassif[index]-Ycontrol[index])
    elif bgFlag=="good" :
        gbIndex= np.where(Yclassif[index] == Ycontrol[index])

    else:
        print 'MakeFitsTable Error: bgFlag not specified.'

    gbTbl = inpTbl.rows(gbIndex[0])
    ap.fitstable.write(gbTbl,fitsFileName,True)
    return 0


def ComputeNaMap(objInfo,nside,a):
    aind = np.where(objInfo[:,2]==a)
    aInfo = objInfo[aind]
    NaMap = ConstructMap(nside,aInfo)
    return NaMap


def ComputeNaFraction(Ntot,Na):
    NaFrac = np.zeros(Na.shape[0])
    ind = np.where(Na<0.)
    NaFrac[ind] = -1.
    ind = np.where((Na>0.)&(Ntot>0.))
    NaFrac[ind] = Na[ind]/Ntot[ind]
    return NaFrac

def ComputeNabMap(objInfo,nside,a,b):
    aind = np.where(objInfo[:,2]==a)
    aInfo = objInfo[aind]
    abind = np.where(aInfo[:,3]==b)
    abInfo = aInfo[abind]
    NabMap = ConstructMap(nside,abInfo)
    return NabMap


def ComputePureMap(NccMap,NabcMap):

    cPureMap = np.zeros(NccMap.shape[0])
    ind = np.where( (NccMap<0.)&(NabcMap<0.) )
    cPureMap[ind] = -1.
    ind = np.where( (NabcMap<0.)&(NccMap>0.) )
    cPureMap[ind] = 1.
    ind = np.where( (NccMap>0.)&(NabcMap>-1) )
    cPureMap[ind] = NccMap[ind]/(NabcMap[ind]+NccMap[ind])
    
    #cPureMap = CopyMap(NccMap)
    #ind = np.where(cPureMap > -1.)
    #cPureMap[ind] = 0.
    #ind = np.where( (NabcMap>-1.)&(NccMap>0.) )
    #cPureMap[ind] = NccMap[ind]/(NabcMap[ind]+NccMap[ind])

    return cPureMap


def ComputeEfficMap(NaMap,NaaMap):

    efficMap = np.zeros(NaMap.shape[0])
    ind = np.where((NaMap<0.)&(NaaMap<0.))
    efficMap[ind] = -1.
    ind = np.where((NaMap>0.)&(NaaMap>0.))
    efficMap[ind] = NaaMap[ind]/(NaMap[ind])

#   efficMap -= 1.
#   ind = np.where( (NaaMap>-1)&(NaMap>0.) )
#   efficMap[ind] = NaaMap[ind]/(NaMap[ind])
    return efficMap

	
