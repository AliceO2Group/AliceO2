/*
Author: Vytautas Vislavicius
Extention of Generic Flow (https://arxiv.org/abs/1312.3572)
*/
#ifndef GFW__H
#define GFW__H
#include "GenericFramework/GFWCumulant.h"
#include <vector>
#include <utility>
#include <algorithm>
#include "TString.h"
#include "TObjArray.h"
using std::vector;

class GFW {
 public:
  struct Region {
    Int_t Nhar, Npar, NpT;
    vector<Int_t> NparVec;
    Double_t EtaMin=-999;
    Double_t EtaMax=-999;
    Int_t BitMask=1;
    TString rName="";
    bool operator<(const Region& a) const {
      return EtaMin < a.EtaMin;
    };
    Region operator=(const Region& a) {
      Nhar=a.Nhar;
      Npar=a.Npar;
      NparVec=a.NparVec;
      NpT =a.NpT;
      EtaMin=a.EtaMin;
      EtaMax=a.EtaMax;
      rName=a.rName;
      BitMask=a.BitMask;
      return *this;
    };
    void PrintStructure() {printf("%s: eta [%f.. %f].",rName.Data(),EtaMin,EtaMax); };
  };
  struct CorrConfig {
    vector<vector<Int_t>> Regs {};
    vector<vector<Int_t>> Hars {};
    vector<Int_t> Overlap;
    Bool_t pTDif=kFALSE;
    TString Head="";
  };
  GFW();
  ~GFW();
  vector<Region> fRegions;
  vector<GFWCumulant> fCumulants;
  vector<Int_t> fEmptyInt;
  void AddRegion(TString refName, Int_t lNhar, Int_t lNpar, Double_t lEtaMin, Double_t lEtaMax, Int_t lNpT=1, Int_t BitMask=1);
  void AddRegion(TString refName, Int_t lNhar, Int_t *lNparVec, Double_t lEtaMin, Double_t lEtaMax, Int_t lNpT=1, Int_t BitMask=1);
  Int_t CreateRegions();
  void Fill(Double_t eta, Int_t ptin, Double_t phi, Double_t weight, Int_t mask, Double_t secondWeight=-1);
  void Clear();// { for(auto ptr = fCumulants.begin(); ptr!=fCumulants.end(); ++ptr) ptr->ResetQs(); };
  GFWCumulant GetCumulant(Int_t index) { return fCumulants.at(index); };
  TComplex Calculate(TString config, Bool_t SetHarmsToZero=kFALSE);
  CorrConfig GetCorrelatorConfig(TString config, TString head = "", Bool_t ptdif=kFALSE);
  TComplex Calculate(CorrConfig corconf, Int_t ptbin, Bool_t SetHarmsToZero, Bool_t DisableOverlap=kFALSE);
 private:
  Bool_t fInitialized;
  void SplitRegions();
  GFWCumulant fEmptyCumulant;
  TComplex TwoRec(Int_t n1, Int_t n2, Int_t p1, Int_t p2, Int_t ptbin, GFWCumulant*, GFWCumulant*, GFWCumulant*);
  TComplex RecursiveCorr(GFWCumulant *qpoi, GFWCumulant *qref, GFWCumulant *qol, Int_t ptbin, vector<Int_t> &hars, vector<Int_t> &pows); //POI, Ref. flow, overlapping region
  TComplex RecursiveCorr(GFWCumulant *qpoi, GFWCumulant *qref, GFWCumulant *qol, Int_t ptbin, vector<Int_t> &hars); //POI, Ref. flow, overlapping region
  //Deprecated and not used (for now):
  void AddRegion(Region inreg) { fRegions.push_back(inreg); };
  Region GetRegion(Int_t index) { return fRegions.at(index); };
  Int_t FindRegionByName(TString refName);
  vector<TString> fCalculatedNames;
  vector<TComplex> fCalculatedQs;
  Int_t FindCalculated(TString identifier);
  //Calculateing functions:
  TComplex Calculate(Int_t poi, Int_t ref, vector<Int_t> hars, Int_t ptbin=0); //For differential, need POI and reference
  TComplex Calculate(Int_t poi, vector<Int_t> hars); //For integrated case
  //Process one string (= one region)
  TComplex CalculateSingle(TString config);

  Bool_t SetHarmonicsToZero(TString &instr);

};
#endif