#ifndef ALIDETECTORPARAM_H
#define ALIDETECTORPARAM_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

////////////////////////////////////////////////
//  Manager class for detector parameters          //
////////////////////////////////////////////////

#include <TNamed.h>
class AliDetectorParam : public TNamed {
public:
  AliDetectorParam();
  virtual Int_t GetNSegmentsTotal() const {return 0;} //get total nuber of segments
  virtual Bool_t Get1DIndex(Int_t */*index*/, const Int_t * /*arrindex*/) {return kFALSE;} 
  //transform multidimensional index to one dimesional
  virtual Bool_t GetNDIndex(const Int_t * /*index1*/, Int_t * /*arrIndex*/) {return kFALSE;}
  //trasnform one dimesional index to multidimesional
  virtual Float_t GetPrimaryLoss(Float_t */*x*/, Int_t */*index*/, Float_t */*angle*/){return 0;}
  virtual Float_t GetTotalLoss(Float_t */*x*/, Int_t */*index*/, Float_t */*angle*/){return 0;}
  virtual void GetClusterSize(Float_t */*x*/, Int_t */*index*/, Float_t */*angle*/, Int_t /*mode*/, Float_t */*sigma*/){;}
  virtual void GetSpaceResolution(Float_t */*x*/, Int_t */*index*/, Float_t */*angle*/, Float_t /*amplitude*/, Int_t /*mode*/, 
				  Float_t */*sigma*/){;}
  virtual Float_t * GetAnglesAccMomentum(Float_t *x, Int_t * index, Float_t* momentum, Float_t *angle); 

  void  SetBField(Float_t b){fBField=b;} //set magnetic field intensity  
  void  SetNPrimLoss(Float_t loss) {fNPrimLoss = loss;}
  void  SetNTotalLoss(Float_t loss) {fNTotalLoss = loss;}
  Float_t GetBFiled() {return fBField;}
  Float_t GetNPrimLoss() {return fNPrimLoss;}
  Float_t GetNTotalLoss() {return fNTotalLoss;}
protected:
  Float_t fBField;  //intensity of magnetic field
  Float_t fNPrimLoss; //number of produced primary  electrons  per cm
  Float_t fNTotalLoss; //total  number of produced  electrons  per cm

  ClassDef(AliDetectorParam,1)  //parameter  object for set:TPC
};




#endif  //ALIDPARAM_H
