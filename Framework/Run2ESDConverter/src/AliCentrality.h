//-*- Mode: C++ -*-
#ifndef ALICentrality_H
#define ALICentrality_H
/* This file is property of and copyright by the ALICE HLT Project        *
 * ALICE Experiment at CERN, All rights reserved.                         *
 * See cxx source for full Copyright notice                               */

//*****************************************************
//   Class AliCentrality
//   author: Alberica Toia
//*****************************************************

#include "TNamed.h"

class AliCentrality : public TNamed
{
 public:

  AliCentrality();  /// constructor
  ~AliCentrality();  /// destructor
  AliCentrality(const AliCentrality& cnt); /// copy constructor
  AliCentrality& operator=(const AliCentrality& cnt);   /// assignment operator

  /// set centrality result
  void SetQuality(Int_t quality) {fQuality = quality;} 
  void SetCentralityV0M(Float_t cent) {fCentralityV0M = cent;} 
  void SetCentralityV0A(Float_t cent) {fCentralityV0A = cent;} 
  void SetCentralityV0A0(Float_t cent) {fCentralityV0A0 = cent;} 
  void SetCentralityV0A123(Float_t cent) {fCentralityV0A123 = cent;} 
  void SetCentralityV0C(Float_t cent) {fCentralityV0C = cent;} 
  void SetCentralityV0A23(Float_t cent) {fCentralityV0A23 = cent;} 
  void SetCentralityV0C01(Float_t cent) {fCentralityV0C01 = cent;} 
  void SetCentralityV0S(Float_t cent) {fCentralityV0S = cent;} 
  void SetCentralityV0MEq(Float_t cent) {fCentralityV0MEq = cent;} 
  void SetCentralityV0AEq(Float_t cent) {fCentralityV0AEq = cent;} 
  void SetCentralityV0CEq(Float_t cent) {fCentralityV0CEq = cent;} 
  void SetCentralityFMD(Float_t cent) {fCentralityFMD = cent;}
  void SetCentralityTRK(Float_t cent) {fCentralityTRK = cent;}
  void SetCentralityTKL(Float_t cent) {fCentralityTKL = cent;}
  void SetCentralityCL0(Float_t cent) {fCentralityCL0 = cent;}
  void SetCentralityCL1(Float_t cent) {fCentralityCL1 = cent;}
  void SetCentralityCND(Float_t cent) {fCentralityCND = cent;}
  void SetCentralityZNA(Float_t cent) {fCentralityZNA = cent;}
  void SetCentralityZNC(Float_t cent) {fCentralityZNC = cent;}
  void SetCentralityZPA(Float_t cent) {fCentralityZPA = cent;}
  void SetCentralityZPC(Float_t cent) {fCentralityZPC = cent;}
  void SetCentralityNPA(Float_t cent) {fCentralityNPA = cent;}
  void SetCentralityV0MvsFMD(Float_t cent) {fCentralityV0MvsFMD = cent;}
  void SetCentralityTKLvsV0M(Float_t cent) {fCentralityTKLvsV0M = cent;}
  void SetCentralityZEMvsZDC(Float_t cent) {fCentralityZEMvsZDC = cent;}

  void SetCentralityV0Mtrue(Float_t cent) {fCentralityV0Mtrue = cent;} 
  void SetCentralityV0Atrue(Float_t cent) {fCentralityV0Atrue = cent;} 
  void SetCentralityV0Ctrue(Float_t cent) {fCentralityV0Ctrue = cent;} 
  void SetCentralityFMDtrue(Float_t cent) {fCentralityFMDtrue = cent;}
  void SetCentralityTRKtrue(Float_t cent) {fCentralityTRKtrue = cent;}
  void SetCentralityTKLtrue(Float_t cent) {fCentralityTKLtrue = cent;}
  void SetCentralityCL0true(Float_t cent) {fCentralityCL0true = cent;}
  void SetCentralityCL1true(Float_t cent) {fCentralityCL1true = cent;}
  void SetCentralityCNDtrue(Float_t cent) {fCentralityCNDtrue = cent;}
  void SetCentralityZNAtrue(Float_t cent) {fCentralityZNAtrue = cent;}
  void SetCentralityZNCtrue(Float_t cent) {fCentralityZNCtrue = cent;}
  void SetCentralityZPAtrue(Float_t cent) {fCentralityZPAtrue = cent;}
  void SetCentralityZPCtrue(Float_t cent) {fCentralityZPCtrue = cent;}

  /// get centrality result
  Float_t GetCentralityPercentile(const char *method) const;
  Int_t   GetCentralityClass10(const char *method) const;
  Int_t   GetCentralityClass5(const char *method) const;
  Bool_t  IsEventInCentralityClass(Float_t a, Float_t b, const char *method) const;

  Float_t GetCentralityPercentileUnchecked(const char *method) const;
  Int_t   GetCentralityClass10Unchecked(const char *method) const;
  Int_t   GetCentralityClass5Unchecked(const char *method) const;
  Bool_t  IsEventInCentralityClassUnchecked(Float_t a, Float_t b, const char *method) const;

  Int_t GetQuality() const;
  void  Reset();

 private:
  Int_t   fQuality; // Quality of centrality determination
  Float_t fCentralityV0M;   // Centrality from V0A+V0C
  Float_t fCentralityV0A;   // Centrality from V0A
  Float_t fCentralityV0A0;  // Centrality from V0A0
  Float_t fCentralityV0A123;// Centrality from V0A123
  Float_t fCentralityV0C;   // Centrality from V0C
  Float_t fCentralityV0A23; // Centrality from V0A23
  Float_t fCentralityV0C01; // Centrality from V0C01
  Float_t fCentralityV0S;   // Centrality from V0S
  Float_t fCentralityV0MEq; // Centrality from V0A+V0C equalized channel
  Float_t fCentralityV0AEq; // Centrality from V0A equalized channel
  Float_t fCentralityV0CEq; // Centrality from V0C equalized channel
  Float_t fCentralityFMD;   // Centrality from FMD
  Float_t fCentralityTRK;   // Centrality from tracks
  Float_t fCentralityTKL;   // Centrality from tracklets
  Float_t fCentralityCL0;   // Centrality from Clusters in layer 0
  Float_t fCentralityCL1;   // Centrality from Clusters in layer 1
  Float_t fCentralityCND;   // Centrality from tracks (candle condition)
  Float_t fCentralityZNA;   // Centrality from ZNA
  Float_t fCentralityZNC;   // Centrality from ZNC
  Float_t fCentralityZPA;   // Centrality from ZPA
  Float_t fCentralityZPC;   // Centrality from ZPC
  Float_t fCentralityNPA;   // Centrality from Npart (MC)
  Float_t fCentralityV0MvsFMD;   // Centrality from V0 vs FMD
  Float_t fCentralityTKLvsV0M;   // Centrality from tracklets vs V0
  Float_t fCentralityZEMvsZDC;   // Centrality from ZEM vs ZDC

  Float_t fCentralityV0Mtrue;   // Centrality from true (sim) V0A+V0C
  Float_t fCentralityV0Atrue;   // Centrality from true (sim) V0A
  Float_t fCentralityV0Ctrue;   // Centrality from true (sim) V0C
  Float_t fCentralityV0MEqtrue; // Centrality from true (sim) V0A+V0C equalized channels
  Float_t fCentralityV0AEqtrue; // Centrality from true (sim) V0A equalized channels
  Float_t fCentralityV0CEqtrue; // Centrality from true (sim) V0C equalized channels
  Float_t fCentralityFMDtrue;   // Centrality from true (sim) FMD
  Float_t fCentralityTRKtrue;   // Centrality from true (sim) tracks
  Float_t fCentralityTKLtrue;   // Centrality from true (sim) tracklets
  Float_t fCentralityCL0true;   // Centrality from true (sim) Clusters in layer 0
  Float_t fCentralityCL1true;   // Centrality from true (sim) Clusters in layer 1
  Float_t fCentralityCNDtrue;   // Centrality from true (sim) tracks (candle condition)
  Float_t fCentralityZNAtrue;   // Centrality from true (sim) ZNA
  Float_t fCentralityZNCtrue;   // Centrality from true (sim) ZNC
  Float_t fCentralityZPAtrue;   // Centrality from true (sim) ZNA
  Float_t fCentralityZPCtrue;   // Centrality from true (sim) ZNC

  ClassDef(AliCentrality, 10)
};
#endif //ALICENTRALITY_H
