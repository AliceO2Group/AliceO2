
/*
Author: Vytautas Vislavicius
Extention of Generic Flow (https://arxiv.org/abs/1312.3572)
*/
#include "GenericFramework/GFWCumulant.h"

GFWCumulant::GFWCumulant():
  fQvector(0),
  fUsed(kBlank),
  fNEntries(-1),
  fN(1),
  fPow(1),
  fPt(1),
  fFilledPts(0),
  fInitialized(kFALSE)
{
};

GFWCumulant::~GFWCumulant()
{
  //printf("Destructor (?) for some reason called?\n");
  //DestroyComplexVectorArray();
};
void GFWCumulant::FillArray(Double_t eta, Int_t ptin, Double_t phi, Double_t weight, Double_t SecondWeight) {
  if(!fInitialized)
    CreateComplexVectorArray(1,1,1);
  if(fPt==1) ptin=0; //If one bin, then just fill it straight; otherwise, if ptin is out-of-range, do not fill
  else if(ptin<0 || ptin>=fPt) return;
  fFilledPts[ptin] = kTRUE;
  for(Int_t lN = 0; lN<fN; lN++) {
    Double_t lSin = TMath::Sin(lN*phi); //No need to recalculate for each power
    Double_t lCos = TMath::Cos(lN*phi); //No need to recalculate for each power
    for(Int_t lPow=0; lPow<PW(lN); lPow++) {
      Double_t lPrefactor = 0;
      //Dont calculate it twice; multiplication is cheaper that power
      //Also, if second weight is specified, then keep the first weight with power no more than 1, and us the other weight otherwise
      //this is important when POIs are a subset of REFs and have different weights than REFs
      if(SecondWeight>0 && lPow>1) lPrefactor = TMath::Power(SecondWeight, lPow-1)*weight;
      else lPrefactor = TMath::Power(weight,lPow);
      Double_t qsin = lPrefactor * lSin;
      Double_t qcos = lPrefactor * lCos;
      fQvector[ptin][lN][lPow](fQvector[ptin][lN][lPow].Re()+qcos,fQvector[ptin][lN][lPow].Im()+qsin);//+=TComplex(qcos,qsin);
    };
  };
  Inc();
};
void GFWCumulant::ResetQs() {
  if(!fNEntries) return; //If 0 entries, then no need to reset. Otherwise, if -1, then just initialized and need to set to 0.
  for(Int_t i=0; i<fPt; i++) {
    fFilledPts[i] = kFALSE;
    for(Int_t lN=0;lN<fN;lN++) {
      for(Int_t lPow=0;lPow<PW(lN);lPow++) {
  	       fQvector[i][lN][lPow](0.,0.);
      };
    };
  };
  fNEntries=0;
};
void GFWCumulant::DestroyComplexVectorArray() {
  if(!fInitialized) return;
  for(Int_t l_n = 0; l_n<fN; l_n++) {
    for(Int_t i=0;i<fPt;i++) {
      delete [] fQvector[i][l_n];
    };
  };
  for(Int_t i=0;i<fPt;i++) {
    delete [] fQvector[i];
  };
  delete [] fQvector;
  delete [] fFilledPts;
  fInitialized=kFALSE;
  fNEntries=-1;
};

void GFWCumulant::CreateComplexVectorArray(Int_t N, Int_t Pow, Int_t Pt) {
  DestroyComplexVectorArray();
  vector<Int_t> pwv;
  for(Int_t i=0;i<N;i++) pwv.push_back(Pow);
  CreateComplexVectorArrayVarPower(N,pwv,Pt);
};
void GFWCumulant::CreateComplexVectorArrayVarPower(Int_t N, vector<Int_t> PowVec, Int_t Pt) {
  DestroyComplexVectorArray();
  fN=N;
  fPow=0;
  fPt=Pt;
  fFilledPts = new Bool_t[Pt];
  fPowVec = PowVec;
  fQvector = new TComplex**[fPt];
  for(Int_t i=0;i<fPt;i++) {
    fQvector[i] = new TComplex*[fN];
  };
  for(Int_t l_n=0;l_n<fN;l_n++) {
    for(Int_t i=0;i<fPt;i++) {
      fQvector[i][l_n] = new TComplex[PW(l_n)];
    };
  };
  ResetQs();
  fInitialized=kTRUE;
};
TComplex GFWCumulant::Vec(Int_t n, Int_t p, Int_t ptbin) {
  if(!fInitialized) return 0;
  if(ptbin>=fPt || ptbin<0) ptbin=0;
  if(n>=0) return fQvector[ptbin][n][p];
  return TComplex::Conjugate(fQvector[ptbin][-n][p]);
};