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
#include "Segmentation.h"

using namespace AliceO2::ITS;

ClassImp(Segmentation)

Segmentation::Segmentation():
fDx(0),
fDz(0),
fDy(0),
fCorr(0){
  // Default constructor
}

Segmentation::~Segmentation(){
  // destructor
  if(fCorr)delete fCorr;
}

void Segmentation::Copy(TObject &obj) const {
  // copy this to obj
  ((Segmentation& ) obj).fDz      = fDz;
  ((Segmentation& ) obj).fDx      = fDx;
  ((Segmentation& ) obj).fDy      = fDy;
  if(fCorr){
    ((Segmentation& ) obj).fCorr    = new TF1(*fCorr); // make a proper copy
  }
  else {
    ((Segmentation& ) obj).fCorr = 0;
  }
}

Segmentation& Segmentation::operator=(
                        const Segmentation &source){
// Operator =
  if(this != &source){
    source.Copy(*this);
  }
  return *this;
}

Segmentation::Segmentation(const Segmentation &source):
    TObject(source),
fDx(0),
fDz(0),
fDy(0),
fCorr(0){
    // copy constructor
  source.Copy(*this);
}
