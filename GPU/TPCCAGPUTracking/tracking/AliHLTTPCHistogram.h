// $Id$
// origin: hough/AliL3Histogram.h,v 1.28 Wed Sep 29 12:01:17 2004 UTC by cvetan

#ifndef ALIHLTTPCHISTOGRAM_H
#define ALIHLTTPCHISTOGRAM_H

#include "AliHLTStdIncludes.h"
#include "AliHLTTPCRootTypes.h"

#include <TH2.h>

class AliHLTTPCHistogram {
 public:

  AliHLTTPCHistogram();
  AliHLTTPCHistogram(Char_t *name,Char_t *id,Int_t nxbin,Double_t xmin,Double_t xmax,Int_t nybin,Double_t ymin,Double_t ymax);
  virtual ~AliHLTTPCHistogram();
  
  void Reset();
  virtual void Fill(Double_t x,Double_t y,Int_t weight=1);
  virtual void Fill(Double_t x,Int_t ybin,Int_t weight=1);
  virtual void Fill(Int_t xbin,Double_t y,Int_t weight=1);
  virtual void Fill(Int_t xbin,Int_t ybin,Int_t weight=1);
  virtual Int_t FindBin(Double_t x,Double_t y) const;
  virtual Int_t FindLabelBin(Double_t x,Double_t y) const;
  virtual Int_t FindXbin(Double_t x) const;
  virtual Int_t FindYbin(Double_t y) const;
  Int_t GetBin(Int_t xbin,Int_t ybin) const;
  Int_t GetLabelBin(Int_t xbin,Int_t ybin) const;
  Int_t GetBinContent(Int_t bin) const;
  void SetBinContent(Int_t xbin,Int_t ybin,Int_t value);
  void SetBinContent(Int_t bin,Int_t value);
  void AddBinContent(Int_t xbin,Int_t ybin,Int_t weight);
  void AddBinContent(Int_t bin,Int_t weight);
  void Add(AliHLTTPCHistogram *h1,Double_t weight=1);
  void SetThreshold(Int_t i) {fThreshold = i;}
  void CreateRootHisto();
  virtual void Draw(Char_t *option="hist");
  virtual void Print() const {};

  friend ofstream& operator<< (ofstream &o, const AliHLTTPCHistogram &h);

  TH2F *GetRootHisto();
    
  Double_t GetXmin() const {return fXmin;}
  Double_t GetXmax() const {return fXmax;}
  Double_t GetYmin() const {return fYmin;}
  Double_t GetYmax() const {return fYmax;}
  virtual Double_t GetBinCenterX(Int_t xbin) const;
  virtual Double_t GetBinCenterY(Int_t ybin) const;
  Double_t GetPreciseBinCenterX(Float_t xbin) const;
  Double_t GetPreciseBinCenterY(Float_t ybin) const;
  Double_t GetBinWidthX() const {return fBinwidthX;}
  Double_t GetBinWidthY() const {return fBinwidthY;}
  Int_t GetFirstXbin() const {return fFirstXbin;}
  Int_t GetLastXbin() const {return fLastXbin;}
  Int_t GetFirstYbin() const {return fFirstYbin;}
  Int_t GetLastYbin() const {return fLastYbin;}
  Int_t GetNbinsX() const {return fNxbins;}
  Int_t GetNbinsY() const {return fNybins;}
  Int_t GetNEntries() const {return fEntries;}

  Int_t *fContent; //!
  Int_t *GetContentArray() const {return fContent;}
    
 protected:
  Char_t fName[100]; // Name of the histogram
  Int_t fNxbins; // Number of bins in the histogram
  Int_t fNybins; // Number of bins in the histogram
  Int_t fNcells; // Overall number of bins in the histogram
  Int_t fEntries; // Number of entries in the histogram
  Int_t fFirstXbin; // First active bin
  Int_t fFirstYbin; // First active bin
  Int_t fLastXbin; // Last active bin
  Int_t fLastYbin; // Last active bin
  Int_t fThreshold; // Bin content threshold 

  Double_t fXmin; // Lower limit in X
  Double_t fYmin; // Lower limit in Y
  Double_t fXmax; // Upper limit in X
  Double_t fYmax; // Upper limit in Y

  TH2F *fRootHisto; // Corresponding ROOT histogram

 private:
  Double_t fBinwidthX; // Bin width of the Hough space
  Double_t fBinwidthY; // Bin width of the Hough space
  
  ClassDef(AliHLTTPCHistogram,1) //2D histogram class
    
};

inline TH2F *AliHLTTPCHistogram::GetRootHisto()
{
  if(!fRootHisto)
    {
      STDCERR<<"AliHLTTPCHistogram::GetRootHisto() : You must first Draw histogram before accessing it"<<STDENDL;
      return 0;
    }
  else
    return fRootHisto;
}

#endif
