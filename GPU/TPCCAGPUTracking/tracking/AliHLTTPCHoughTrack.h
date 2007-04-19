// @(#) $Id: AliHLTHoughTrack.h,v 1.1 2006/11/30 17:45:43 hristov Exp 
// origin hough/AliL3HoughTrack.h,v 1.8 Thu Mar 31 04:48:58 2005 UTC by cvetan

#ifndef ALIHLTTPCHOUGHTRACK_H
#define ALIHLTTPCHOUGHTRACK_H

#include "AliHLTTrack.h"

class AliHLTTPCHoughTrack : public AliHLTTrack {
  
 public:
  AliHLTTPCHoughTrack(); 
  virtual ~AliHLTTPCHoughTrack();
  
  virtual void Set(AliHLTTrack *track);
  virtual Int_t Compare(const AliHLTTrack *track) const;
  
  Bool_t IsHelix() const {return fIsHelix;}
  void UpdateToFirstRow();
  void SetTrackParameters(Double_t kappa,Double_t eangle,Int_t weight);  
  void SetTrackParametersRow(Double_t alpha1,Double_t alpha2,Double_t eta,Int_t weight);  
  void SetLineParameters(Double_t psi,Double_t D,Int_t weight,Int_t *rowrange,Int_t refrow);

  Int_t GetWeight()  const {return fWeight;}
  Double_t GetPsiLine() const {return fPsiLine;}
  Double_t GetDLine() const {return fDLine;}

  Int_t GetEtaIndex() const {return fEtaIndex;}
  Double_t GetEta() const {return fEta;}
  Int_t GetSlice()  const {return fSlice;}
  void GetLineCrossingPoint(Int_t padrow,Float_t *xy);

  Float_t    GetBinX()   const {return fBinX;}
  Float_t    GetBinY()   const {return fBinY;}
  Float_t    GetSizeX()  const {return fSizeX;}
  Float_t    GetSizeY()  const {return fSizeY;}
  
  void SetHelixTrue() {fIsHelix=kTRUE;}
  void SetSlice(Int_t slice) {fSlice=slice;}
  void SetEta(Double_t f);
  void SetWeight(Int_t i,Bool_t update=kFALSE) {if(update) fWeight+= i; else fWeight = i;}
  void SetEtaIndex(Int_t f) {fEtaIndex = f;}
  void SetBestMCid(Int_t f,Double_t mindist);
  void SetDLine(Double_t f) {fDLine=f;}
  void SetPsiLine(Double_t f) {fPsiLine=f;}

  void SetBinXY(Float_t binx,Float_t biny,Float_t sizex,Float_t sizey) {fBinX = binx; fBinY = biny; fSizeX = sizex; fSizeY = sizey;}

 private:
  
  Double_t fMinDist;//Minimum distance to a generated track while associating mc label 
  Int_t fWeight;//Track weight
  Int_t fEtaIndex;//Eta slice index
  Double_t fEta;//Track Eta
  Int_t fSlice; //The slice where this track was found

  Double_t fDLine;//??
  Double_t fPsiLine;//??
 
  Bool_t fIsHelix;//Is the track helix or not?

  Float_t fBinX,fBinY,fSizeX,fSizeY;//Size and position of the hough space peak 

  ClassDef(AliHLTTPCHoughTrack,1) //Track class for Hough tracklets

};

#endif
