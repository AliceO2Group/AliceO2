/*
Author: Vytautas Vislavicius
Extention of Generic Flow (https://arxiv.org/abs/1312.3572)
*/
#ifndef GFWCUMULANT__H
#define GFWCUMULANT__H
#include "TComplex.h"
#include "TNamed.h"
#include "TMath.h"
#include "TAxis.h"
using std::vector;
class GFWCumulant {
 public:
  GFWCumulant();
  ~GFWCumulant();
  void ResetQs();
  void FillArray(Double_t eta, Int_t ptin, Double_t phi, Double_t weight=1, Double_t SecondWeight=-1);
  enum UsedFlags_t {kBlank = 0, kFull=1, kPt=2};
  void SetType(UInt_t infl) { DestroyComplexVectorArray(); fUsed = infl; };
  void Inc() { fNEntries++; };
  Int_t GetN() { return fNEntries; };
  // protected:
  TComplex ***fQvector;
  UInt_t fUsed;
  Int_t fNEntries;
  //Q-vectors. Could be done recursively, but maybe defining each one of them explicitly is easier to read
  TComplex Vec(Int_t, Int_t, Int_t ptbin=0); //envelope class to summarize pt-dif. Q-vec getter
  Int_t fN; //! Harmonics
  Int_t fPow; //! Power
  vector<Int_t> fPowVec; //! Powers array
  Int_t fPt; //!fPt bins
  Bool_t *fFilledPts;
  Bool_t fInitialized; //Arrays are initialized
  void CreateComplexVectorArray(Int_t N=1, Int_t P=1, Int_t Pt=1);
  void CreateComplexVectorArrayVarPower(Int_t N=1, vector<Int_t> Pvec={1}, Int_t Pt=1);
  Int_t PW(Int_t ind) { return fPowVec.at(ind); }; //No checks to speed up, be carefull!!!
  void DestroyComplexVectorArray();
  Bool_t IsPtBinFilled(Int_t ptb) { if(!fFilledPts) return kFALSE; return fFilledPts[ptb]; };
};

#endif