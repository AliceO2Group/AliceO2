
// -*- mode: C++ -*- 
#ifndef ALIESDFIT_H
#define ALIESDFIT_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


//-------------------------------------------------------------------------
//                          Class AliESDFIT
//   This is a class that summarizes the FIT data for the ESD   
//   Origin: Christian Klein-Boesing, CERN, Christian.Klein-Boesing@cern.ch 
//-------------------------------------------------------------------------



#include <TObject.h>

class AliESDFIT: public TObject {
public:
  AliESDFIT();
  AliESDFIT(const AliESDFIT& tzero);
  AliESDFIT& operator=(const AliESDFIT& tzero);
  virtual void Copy(TObject &obj) const;

  Float_t GetFITzVertex() const {return fFITzVertex;}
  void SetFITzVertex(Float_t z) {fFITzVertex=z;}
  //1st
  Float_t GetFITT0(Int_t i) const {return fT0[i];}
  const Float_t * GetFITT0() const {return fT0;}
  void SetFITT0(Int_t icase, Float_t time) { fT0[icase] = time;}
  //best
  Float_t GetT0best(Int_t i) const {return fT0best[i];}
  const Float_t * GetT0best() const {return fT0best;}
  void SetT0best(Int_t icase, Float_t time) { fT0best[icase] = time;}

  const Float_t * GetFITtime() const {return fFITtime;}
  void SetFITtime(Float_t time[288]) {
    for (Int_t i=0; i<288; i++) fFITtime[i] = time[i];
  }
  const Float_t * GetFITamplitude() const {return fFITamplitude;}
  void SetFITamplitude(Float_t amp[288]) {
    for (Int_t i=0; i<288; i++) fFITamplitude[i] = amp[i];
  }
  const Float_t * GetFITphotons() const {return fFITphotons;}
  void SetFITphotons(Float_t amp[288]) {
    for (Int_t i=0; i<288; i++) fFITphotons[i] = amp[i];
  }

  void    Reset();
  void    Print(const Option_t *opt=0) const;


private:

  Float_t   fT0[3];     // interaction time in ps with 1st time( A&C, A, C)
  Float_t   fFITzVertex;       // vertex z position estimated by the T0, cm
  Float_t   fFITtime[288];      // best TOF on each T0 PMT
  Float_t   fFITamplitude[288]; // number of particles(MIPs) on each T0 PMT
  Float_t   fFITphotons[288]; // number of particles(MIPs) on each T0 PMT
  Float_t   fT0best[3]; // interaction time in ps ( A&C, A, C) with best time
  
  ClassDef(AliESDFIT,3)
};


#endif
