/*
Author: Vytautas Vislavicius
Extention of Generic Flow (https://arxiv.org/abs/1312.3572)
*/
#ifndef FLOWCONTAINER__H
#define FLOWCONTAINER__H
#include "TH3F.h"
#include "TProfile2D.h"
#include "TProfile.h"
#include "TNamed.h"
#include "TH1.h"
#include "TMath.h"
#include "TFile.h"
#include "TAxis.h"
#include "TString.h"
#include "TObjArray.h"
#include "GenericFramework/ProfileSubset.h"
#include "Framework/HistogramRegistry.h"
#include "TRandom.h"
#include "TString.h"
#include "TCollection.h"
#include "TAxis.h"

class FlowContainer:public TNamed {
 public:
  FlowContainer();
  FlowContainer(const char *name);
  ~FlowContainer();
  enum StatisticsType {kSingleSample, kJackKnife, kBootstrap};
  void Initialize(TObjArray *inputList, const o2::framework::AxisSpec axis, Int_t nRandomized=0);
  void Initialize(TObjArray *inputList, Int_t nMultiBins, Double_t MultiMin, Double_t MultiMax, Int_t nRandomized=0);
  Bool_t CreateBinsFromAxis(TAxis *inax);
  void SetXAxis(TAxis *inax);
  void SetXAxis();
  void RebinMulti(Int_t rN) { if(fProf) fProf->RebinX(rN); };
  Int_t GetNMultiBins() { return fProf->GetNbinsX(); };
  Double_t GetMultiAtBin(Int_t bin) { return fProf->GetXaxis()->GetBinCenter(bin); };
  Int_t FillProfile(const char *hname, Double_t multi, Double_t y, Double_t w, Double_t rn);
  TProfile2D *GetProfile() { return fProf; };
  void OverrideProfileErrors(TProfile2D *inpf);
  void ReadAndMerge(const char *infile);
  void PickAndMerge(TFile *tfi);
  Bool_t OverrideBinsWithZero(Int_t xb1, Int_t yb1, Int_t xb2, Int_t yb2);
  Bool_t OverrideMainWithSub(Int_t subind, Bool_t ExcludeChosen);
  Bool_t RandomizeProfile(Int_t nSubsets=0);
  Bool_t CreateStatisticsProfile(StatisticsType StatType, Int_t arg);
  TObjArray *GetSubProfiles() { return fProfRand; };
  Long64_t Merge(TCollection *collist);
  void SetIDName(TString newname); //! do not store
  void SetPtRebin(Int_t newval) { fPtRebin=newval; };
  void SetPtRebin(Int_t nbins, Double_t *binedges);
  void SetMultiRebin(Int_t nbins, Double_t *binedges);
  Double_t *GetMultiRebin(Int_t &nBins);
  void SetPropagateErrors(Bool_t newval) { fPropagateErrors = newval; };
  TProfile *GetCorrXXVsMulti(const char *order, Int_t l_pti=0);//pti = 0 for pt-integrated
  TProfile *GetCorrXXVsPt(const char *order, Double_t lminmulti=-1, Double_t lmaxmulti=-1); //0 for multi. integrated
  TH1D *GetHistCorrXXVsMulti(const char *order, Int_t l_pti=0);//pti = 0 for pt-integrated
  TH1D *GetHistCorrXXVsPt(const char *order, Double_t lminmulti=-1, Double_t lmaxmulti=-1); //0 for multi. integrated

  TH1D *GetVN2VsMulti(Int_t n=2, Int_t l_pta=0) { return GetVN2VsX(n,kFALSE,l_pta);};
  TH1D *GetVN2VsPt(Int_t n=2, Double_t min=-1, Double_t max=-1) { return GetVN2VsX(n,kTRUE,min,max);};
  TH1D *GetCN4VsMulti(Int_t n=2, Int_t pti=0) { return GetCN4VsX(n,kFALSE,pti); };
  TH1D *GetCN4VsPt(Int_t n=2, Double_t min=-1, Double_t max=-1) { return GetCN4VsX(n,kTRUE,min,max); };

  TH1D *GetVN4VsMulti(Int_t n=2, Int_t pti=0) { return GetVN4VsX(n,kFALSE,pti); };
  TH1D *GetVN4VsPt(Int_t n=2, Double_t min=-1, Double_t max=-1) { return GetVN4VsX(n,kTRUE,min,max); };

  TH1D *GetVN6VsMulti(Int_t n=2, Int_t pti=0) { return GetVN6VsX(n,kFALSE,pti); };
  TH1D *GetVN6VsPt(Int_t n=2, Double_t min=-1, Double_t max=-1) { return GetVN6VsX(n,kTRUE,min,max); };

  TH1D *GetVN8VsMulti(Int_t n=2, Int_t pti=0) { return GetVN8VsX(n,kFALSE,pti); };
  TH1D *GetVN8VsPt(Int_t n=2, Double_t min=-1, Double_t max=-1) { return GetVN8VsX(n,kTRUE,min,max); };

  TH1D *GetCNN(Int_t n=2, Int_t c=2, Bool_t onPt=kTRUE, Double_t arg1=-1, Double_t arg2=-1);
  TH1D *GetVNN(Int_t n=2, Int_t c=2, Bool_t onPt=kTRUE, Double_t arg1=-1, Double_t arg2=-1);


  // private:

  Double_t CN2Value(Double_t cor2); //This is redundant, but adding for completeness
  Double_t CN2Error(Double_t cor2e); //Also redundant
  Double_t VN2Value(Double_t cor2);
  Double_t VN2Error(Double_t cor2, Double_t cor2e);
  Double_t VDN2Value(Double_t cor2d, Double_t cor2);
  Double_t VDN2Error(Double_t cor2d, Double_t cor2de, Double_t cor2, Double_t cor2e);

  Double_t CN4Value(Double_t cor4, Double_t cor2);
  Double_t CN4Error(Double_t cor4e, Double_t cor2, Double_t cor2e);
  Double_t DN4Value(Double_t cor4d, Double_t cor2d, Double_t cor2);
  Double_t DN4Error(Double_t cor4de, Double_t cor2d, Double_t cor2de, Double_t cor2, Double_t cor2e);
  Double_t VN4Value(Double_t c4);
  Double_t VN4Error(Double_t c4, Double_t c4e);
  Double_t VDN4Value(Double_t d4, Double_t c4);
  Double_t VDN4Error(Double_t d4, Double_t d4e, Double_t c4, Double_t c4e);

  Double_t CN6Value(Double_t cor6, Double_t cor4, Double_t cor2);
  Double_t CN6Error(Double_t cor6e, Double_t cor4, Double_t cor4e,  Double_t cor2, Double_t cor2e);

  Double_t DN6Value(Double_t cor6d, Double_t cor4d, Double_t cor2d, Double_t cor4, Double_t cor2);
  Double_t DN6Error(Double_t d6e, Double_t d4, Double_t d4e, Double_t d2,
		    Double_t d2e, Double_t c4, Double_t c4e, Double_t c2,
		    Double_t c2e);
  Double_t VN6Value(Double_t c6);
  Double_t VN6Error(Double_t c6, Double_t c6e);
  Double_t VDN6Value(Double_t d6, Double_t c6);
  Double_t VDN6Error(Double_t d6, Double_t d6e, Double_t c6, Double_t c6e);

  Double_t CN8Value(Double_t cor8, Double_t cor6, Double_t cor4, Double_t cor2);
  Double_t CN8Error(Double_t cor8e, Double_t cor6, Double_t cor6e,
		    Double_t cor4, Double_t cor4e, Double_t cor2, Double_t cor2e);
  Double_t DN8Value(Double_t cor8d, Double_t cor6d, Double_t cor4d, Double_t cor2d, Double_t cor6, Double_t cor4, Double_t cor2);
  Double_t DN8Error(Double_t d8e, Double_t d6, Double_t d6e, Double_t d4,
		    Double_t d4e, Double_t d2, Double_t d2e, Double_t c6,
		    Double_t c6e, Double_t c4, Double_t c4e, Double_t c2,
		    Double_t c2e);
  Double_t VN8Value(Double_t c8);
  Double_t VN8Error(Double_t c8, Double_t c8e);
  Double_t VDN8Value(Double_t d8, Double_t c8);
  Double_t VDN8Error(Double_t d8, Double_t d8e, Double_t c8, Double_t c8e);

  TH1D *GetCN2VsX(Int_t n=2, Bool_t onPt=kTRUE, Double_t larg1=-1, Double_t larg2=-1); //This one is redundant
  TH1D *GetVN2VsX(Int_t n=2, Bool_t onPt=kTRUE, Double_t larg1=-1, Double_t larg2=-1);
  TH1D *GetCN4VsX(Int_t n=2, Bool_t onPt=kTRUE, Double_t larg1=-1, Double_t larg2=-1);
  TH1D *GetVN4VsX(Int_t n=2, Bool_t onPt=kTRUE, Double_t larg1=-1, Double_t larg2=-1);
  TH1D *GetCN6VsX(Int_t n=2, Bool_t onPt=kTRUE, Double_t larg1=-1, Double_t larg2=-1);
  TH1D *GetVN6VsX(Int_t n=2, Bool_t onPt=kTRUE, Double_t larg1=-1, Double_t larg2=-1);
  TH1D *GetCN8VsX(Int_t n=2, Bool_t onPt=kTRUE, Double_t larg1=-1, Double_t larg2=-1);
  TH1D *GetVN8VsX(Int_t n=2, Bool_t onPt=kTRUE, Double_t larg1=-1, Double_t larg2=-1);


  TH1D *GetVN2(TH1D *cn2);
  TH1D *GetVN4(TH1D *inh);
  TH1D *GetVN6(TH1D *inh);
  TH1D *GetVN8(TH1D *inh);
  TH1D *GetCN2(TH1D *corrN2);
  TH1D *GetCN4(TH1D *corrN4, TH1D *corrN2);
  TH1D *GetCN6(TH1D *corrN6, TH1D *corrN4, TH1D *corrN2);
  TH1D *GetCN8(TH1D *corrN8, TH1D *corrN6, TH1D *corrN4, TH1D *corrN2);
  TH1D *ProfToHist(TProfile *inpf);
  TProfile2D *fProf;
  TObjArray *fProfRand;
  Int_t fNRandom;
  TString fIDName;
  Int_t fPtRebin; //! do not store
  Double_t *fPtRebinEdges; //! do not store
  Int_t fMultiRebin; //! do not store
  Double_t *fMultiRebinEdges; //! do not store
  TAxis *fXAxis;
  Int_t fNbinsPt; //! Do not store; stored in the fXAxis
  Double_t *fbinsPt; //! Do not store; stored in fXAxis
  Bool_t fPropagateErrors; //! do not store
  TProfile *GetRefFlowProfile(const char *order, Double_t m1=-1, Double_t m2=-1);
  ClassDef(FlowContainer, 2);
};


#endif