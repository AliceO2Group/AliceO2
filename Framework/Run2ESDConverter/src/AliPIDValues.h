#ifndef ALI_PID_VALUES_H
#define ALI_PID_VALUES_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//---------------------------------------------------------------//
//        Base class for handling the pid response               //
//        functions of all detectors                             //
//        and give access to the nsigmas                         //
//                                                               //
//   Origin: Jens Wiechula, Uni Tuebingen, jens.wiechula@cern.ch //
//---------------------------------------------------------------//

#include <TObject.h>

#include "AliPID.h"
#include "AliPIDResponse.h"

class AliPIDValues : public TObject {
public:
  AliPIDValues();
  AliPIDValues(const AliPIDValues &val);
  AliPIDValues(Double_t val[], Int_t nspecies, AliPIDResponse::EDetPidStatus status=AliPIDResponse::kDetPidOk);
  
  AliPIDValues& operator= (const AliPIDValues &val);
  void Copy(TObject &obj) const;
  
  void SetValues(const Double_t val[], Int_t nspecies, AliPIDResponse::EDetPidStatus status=AliPIDResponse::kDetPidOk);
  AliPIDResponse::EDetPidStatus GetValues(Double_t val[], Int_t nspecies) const;

  Double_t GetValue(AliPID::EParticleType type) const;

  void SetPIDStatus(AliPIDResponse::EDetPidStatus status) { fPIDStatus=status; }
  AliPIDResponse::EDetPidStatus GetPIDStatus() const { return fPIDStatus; }
  
private:
  Double32_t fValues[AliPID::kSPECIESCN];    //[0.,0.,8] PID values
  AliPIDResponse::EDetPidStatus fPIDStatus;  //PID status of the detector

  ClassDef(AliPIDValues,1);                  //Store PID values for each particle type
};

#endif
