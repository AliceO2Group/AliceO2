////////////////////////////////////////////////
//  Segmentation class for set:ITS            //
//  All methods implemented in the derived    //
//  classes are set = 0 in the header file    //
//  so this class cannot be instantiated      //
//  methods implemented in a part of the      //
// derived classes are implemented here as    //
// TObject::MayNotUse                         // 
////////////////////////////////////////////////

#include <TF1.h>
#include "AliITSsegmentation.h"

ClassImp(AliITSsegmentation)

AliITSsegmentation::AliITSsegmentation():
fDx(0),
fDz(0),
fDy(0),
fCorr(0){
  // Default constructor
}

AliITSsegmentation::~AliITSsegmentation(){
  // destructor
  if(fCorr)delete fCorr;
}

void AliITSsegmentation::Copy(TObject &obj) const {
  // copy this to obj
  ((AliITSsegmentation& ) obj).fDz      = fDz;
  ((AliITSsegmentation& ) obj).fDx      = fDx;
  ((AliITSsegmentation& ) obj).fDy      = fDy;
  if(fCorr){
    ((AliITSsegmentation& ) obj).fCorr    = new TF1(*fCorr); // make a proper copy
  }
  else {
    ((AliITSsegmentation& ) obj).fCorr = 0;
  }
}

AliITSsegmentation& AliITSsegmentation::operator=(
                        const AliITSsegmentation &source){
// Operator =
  if(this != &source){
    source.Copy(*this);
  }
  return *this;
}

AliITSsegmentation::AliITSsegmentation(const AliITSsegmentation &source):
    TObject(source),
fDx(0),
fDz(0),
fDy(0),
fCorr(0){
    // copy constructor
  source.Copy(*this);
}
