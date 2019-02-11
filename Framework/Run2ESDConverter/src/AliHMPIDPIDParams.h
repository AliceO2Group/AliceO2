#ifndef ALIHMPIDPIDPARAMS_H
#define ALIHMPIDPIDPARAMS_H
/* Copyright(c) 1998-2010, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//***********************************************************
// Class AliHMPIDPIDparams
// class to store PID parameters for HMPID in OADB
// Author: G. Volpe, giacomo.volpe@cern.ch
//***********************************************************

#include <TNamed.h>

class TObjArray;

class AliHMPIDPIDParams : public TNamed {

 public:
  AliHMPIDPIDParams();
  AliHMPIDPIDParams(Char_t * name);
  AliHMPIDPIDParams& operator= (const AliHMPIDPIDParams& c);
  AliHMPIDPIDParams(const AliHMPIDPIDParams& c);
  virtual ~AliHMPIDPIDParams();

  TObjArray*  GetHMPIDrefIndex() const     {return fHMPIDRefIndexArray;}  
  void SetHMPIDrefIndex(TObjArray *array)  {fHMPIDRefIndexArray = array;}

 private:
  TObjArray   *fHMPIDRefIndexArray;                           // C6F14 refractive index

  ClassDef(AliHMPIDPIDParams,1);

};

#endif

