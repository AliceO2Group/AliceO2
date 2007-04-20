// @(#) $Id$
// origin: hough/AliL3Histogram1D.h,v 1.5 Thu Jun 17 13:18:42 2004 UTC by cvetan 

#ifndef ALIHLTTPCHISTOGRAM1D_H
#define ALIHLTTPCHISTOGRAM1D_H

#include "AliHLTTPCRootTypes.h"

class TH1F;

class AliHLTTPCHistogram1D {
  
 public:
  AliHLTTPCHistogram1D();
  AliHLTTPCHistogram1D(Char_t *name,Char_t *id,Int_t nxbin,Double_t xmin,Double_t xmax);
  virtual ~AliHLTTPCHistogram1D();
  
  void Reset();
  void Fill(Double_t x,Int_t weight=1);
  void AddBinContent(Int_t bin,Int_t weight);
  Int_t GetMaximumBin() const;
  Int_t FindBin(Double_t x) const;
  Double_t GetBinContent(Int_t bin) const;
  Double_t GetBinCenter(Int_t bin) const;
  Int_t GetNEntries() const {return fEntries;}
  
  void SetBinContent(Int_t bin,Int_t value);
  void SetThreshold(Int_t i) {fThreshold = i;}
  
  void Draw(Char_t *option="hist");
  TH1F *GetRootHisto() {return fRootHisto;}
  
 private:
  
  Double_t *fContent; //!
  Char_t fName[100];//Histogram title
  Int_t fNbins;//Number of bins
  Int_t fNcells;//Number of cells
  Int_t fEntries;//Number of entries

  Int_t fThreshold;//Bin content threshold
  Double_t fXmin;//Lower limit in X
  Double_t fXmax;//Upper limit in X

  
  TH1F *fRootHisto;//The corresponding ROOT histogram

  ClassDef(AliHLTTPCHistogram1D,1) //1D histogram class
    
};

#endif
