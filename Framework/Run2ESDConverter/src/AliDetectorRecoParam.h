#ifndef ALIDETECTORRECOPARAM_H
#define ALIDETECTORRECOPARAM_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Base Class for Detector reconstruction parameters                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////


#include "TNamed.h"

#include "AliRecoParam.h"

class AliDetectorRecoParam : public TNamed
{
  
 public: 
  AliDetectorRecoParam();
  virtual ~AliDetectorRecoParam();
  void  Print(Option_t */*option*/) const {Dump();}

  Int_t          GetEventSpecie() const { return fEventSpecie; }
  void           SetEventSpecie(AliRecoParam::EventSpecie_t specie) { fEventSpecie = specie; }
  void           SetAsDefault() { fEventSpecie |= AliRecoParam::kDefault; }
  Bool_t         IsDefault() const { return (fEventSpecie & AliRecoParam::kDefault); }

private:

  Int_t fEventSpecie; // Event specie for which the reco-param object is valid
  
  ClassDef(AliDetectorRecoParam, 3)
};


#endif
