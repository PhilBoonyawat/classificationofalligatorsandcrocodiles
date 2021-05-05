import botnoi as bn
import pickle
from botnoi import cv
import numpy as np

### load model
modfile = '/Users/boonyawatariyawatkul/Desktop/botnoi-cv-detect/mymod.mod'
pickle.load(open(modfile,'rb'))
def predictimg(imgurl):
  a = cv.image(imgurl)
  feat = a.getresnet50()
  res = model.predict([feat])
  return res
