#ifndef ALIMATHBASE_H
#define ALIMATHBASE_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */


 
#include "TObject.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TGraph2D.h"
#include "TGraph.h"

class TH1F;
class TH3;

 
class AliMathBase : public TObject
{
 public:
  AliMathBase();
  virtual ~AliMathBase();
  static void    EvaluateUni(Int_t nvectors, Double_t *data, Double_t &mean, Double_t &sigma, Int_t hh);
  static void    EvaluateUniExternal(Int_t nvectors, Double_t *data, Double_t &mean, Double_t &sigma, Int_t hh, Float_t externalfactor=1);
  static Int_t  Freq(Int_t n, const Int_t *inlist, Int_t *outlist, Bool_t down);    
  static void TruncatedMean(TH1F * his, TVectorD *param, Float_t down=0, Float_t up=1.0, Bool_t verbose=kFALSE);
  static void LTM(TH1F * his, TVectorD *param=0 , Float_t fraction=1,  Bool_t verbose=kFALSE);
  static Double_t  FitGaus(TH1F* his, TVectorD *param=0, TMatrixD *matrix=0, Float_t xmin=0, Float_t xmax=0,  Bool_t verbose=kFALSE);
  static Double_t  FitGaus(Float_t *arr, Int_t nBins, Float_t xMin, Float_t xMax, TVectorD *param=0, TMatrixD *matrix=0, Bool_t verbose=kFALSE);
  static Float_t  GetCOG(Short_t *arr, Int_t nBins, Float_t xMin, Float_t xMax, Float_t *rms=0, Float_t *sum=0);

  static Double_t TruncatedGaus(Double_t mean, Double_t sigma, Double_t cutat);
  static Double_t TruncatedGaus(Double_t mean, Double_t sigma, Double_t leftCut, Double_t rightCut);

  static TGraph2D *  MakeStat2D(TH3 * his, Int_t delta0, Int_t delta1, Int_t type);
  static TGraph *  MakeStat1D(TH3 * his, Int_t delta1, Int_t type);

  static Double_t ErfcFast(Double_t x);                           // Complementary error function erfc(x)
  static Double_t ErfFast(Double_t x) {return 1-ErfcFast(x);}     // Error function erf(x)

  //
  // TestFunctions:
  //

  //
  // Bethe-Bloch formula parameterizations
  //
  static Double_t BetheBlochAleph(Double_t bg,
                                  Double_t kp1=0.76176e-1,
                                  Double_t kp2=10.632,
                                  Double_t kp3=0.13279e-4,
                                  Double_t kp4=1.8631,
                                  Double_t kp5=1.9479
				  );
    
 ClassDef(AliMathBase,0) // Various mathematical tools for physics analysis - which are not included in ROOT TMath
 
};
#endif
