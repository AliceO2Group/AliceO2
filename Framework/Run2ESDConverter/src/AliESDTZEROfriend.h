// -*- mode: C++ -*- 
#ifndef ALIESDTZEROFRIEND_H
#define ALIESDTZEROFRIEND_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

////////////////////////////////////////////////////////////////
///////////////
///
/// This is a class for containing time coorected by SPD vertex and amplitude
/// It is written to the ESD-friend file
///
///////////////////////////////////////////////////////////////////////////////

#include <TObject.h>

class AliESDTZEROfriend: public TObject {
  public :
    AliESDTZEROfriend();
    virtual ~AliESDTZEROfriend(){;}

    AliESDTZEROfriend(const AliESDTZEROfriend& tzerofriend);
    AliESDTZEROfriend& operator = (const AliESDTZEROfriend& tzerofriend);

    virtual void Copy(TObject &obj) const;
    void Reset();

// Getters & setters 

   Double32_t * GetT0timeCorr()  {return fT0time;}
  void SetT0timeCorr(Double32_t time[24]) {
  for (Int_t i=0; i<24; i++) fT0time[i] = time[i];
  }
  
  Double32_t * GetT0ampQTC()  {return fT0ampQTC;}
  void SetT0ampQTC(Double32_t amp[24]) {
    for (Int_t i=0; i<24; i++) fT0ampQTC[i] = amp[i];
  }

  Double32_t * GetT0ampLEDminCFD() {return fT0ampLEDminCFD;}
  void SetT0ampLEDminCFD(Double32_t amp[24]) {
    for (Int_t i=0; i<24; i++) fT0ampLEDminCFD[i] = amp[i];
  }

 private:

  Double32_t   fT0time[24];      // best TOF on each T0 PMT
  Double32_t   fT0ampQTC[24]; // amp in #channels QTC on each T0 PMT
  Double32_t   fT0ampLEDminCFD[24]; // amp in #channels LED-CFD on each T0 PMT

  ClassDef(AliESDTZEROfriend,1)
};

#endif
