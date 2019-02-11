#ifndef ALITOFTRIGGERMASK_H
#define ALITOFTRIGGERMASK_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */
/* $Id$ */

// *
// *
// *
// * this class defines the TOF object to be stored
// * in OCDB on a run-by-run basis in order to have the status
// * of TOF trigger inputs. it stores 32 bit masks for each crate
// * 
// *
// *
// *

#include "TObject.h"
#include "TMath.h"
#include "TH2F.h"

class AliTOFTriggerMask :
public TObject
{

 public:

  AliTOFTriggerMask(); // default constructor
  virtual ~AliTOFTriggerMask(); // default destructor
  AliTOFTriggerMask(const AliTOFTriggerMask &source); // copy constructor
  AliTOFTriggerMask &operator=(const AliTOFTriggerMask &source); // operator=

  UInt_t GetTriggerMask(UInt_t icrate) const {return icrate < 72 ? fTriggerMask[icrate] : 0;}; // get trigger mask
  UInt_t *GetTriggerMaskArray() {return fTriggerMask;}; // get trigger mask array

  void SetTriggerMaskArray(UInt_t *array); // set trigger mask array

  void ResetMask();
  void SetON(Int_t icrate,Int_t ich);
  Bool_t IsON(Int_t icrate,Int_t ich);

  Int_t GetNumberMaxiPadOn(); // return number of active MP
  TH2F *GetHistoMask(); // return active MP map

 private:
  static Int_t fPowerMask[24]; // 2-power mask table

  UInt_t fTriggerMask[72]; //[0,16777215,23] trigger mask array

  ClassDef(AliTOFTriggerMask, 2);
};

#endif /* ALITOFTRIGGERMASK_H */
