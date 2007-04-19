// @(#) $Id$
// origin: hough/AliL3HoughEval.h,v 1.17 Thu Jun 17 10:36:14 2004 UTC by cvetan

#ifndef ALIHLTTPCHOUGHEVAL_H
#define ALIHLTTPCHOUGHEVAL_H

#include "AliHLTTPCRootTypes.h"

class AliHLTTPCTrackArray;
class AliHLTTPCHoughTransformer;
class AliHLTTPCHoughTrack;
class AliHLTTPCDigitRowData;
class AliHLTTPCHistogram;
class AliHLTTPCHistogram1D;

class AliHLTTPCHoughEval {
  
 public:
  AliHLTTPCHoughEval(); 
  virtual ~AliHLTTPCHoughEval();
  
  void InitTransformer(AliHLTTPCHoughTransformer *transformer);
  void GenerateLUT();
  void DisplayEtaSlice(Int_t etaindex,AliHLTTPCHistogram *hist);
  Bool_t LookInsideRoad(AliHLTTPCHoughTrack *track,Int_t &nrowscrossed,Int_t *rowrange,Bool_t remove=kFALSE);
  void CompareMC(AliHLTTPCTrackArray *tracks,Char_t *goodtracks="good_tracks",Int_t treshold=0);
  void FindEta(AliHLTTPCTrackArray *tracks);
  
  //Getters
  AliHLTTPCHistogram1D *GetEtaHisto(Int_t i) {if(!fEtaHistos) return 0; if(!fEtaHistos[i]) return 0; return fEtaHistos[i];}

  //Setters:
  void RemoveFoundTracks() {fRemoveFoundTracks = kTRUE;}
  void SetNumOfRowsToMiss(Int_t i) {fNumOfRowsToMiss = i;}
  void SetNumOfPadsToLook(Int_t i) {fNumOfPadsToLook = i;}
  void SetSlice(Int_t i) {fSlice=i;}
  void SetZVertex(Float_t zvertex) {fZVertex=zvertex;}

 private:

  Int_t fSlice;//Index of the slice being processed
  Int_t fPatch;//Index of the patch being processed
  Int_t fNrows;//Number of padrows inside the patch
  Int_t fNEtaSegments;//Number of eta slices
  Double_t fEtaMin;//Minimum allowed eta
  Double_t fEtaMax;//Maximum allowed eta
  Int_t fNumOfPadsToLook;//Padrow search window
  Int_t fNumOfRowsToMiss;//Maximum numbers of padrow which could be missed
  AliHLTTPCHistogram1D **fEtaHistos; //!
  Float_t fZVertex;//Z position of the primary vertex

  //Flags
  Bool_t fRemoveFoundTracks;//Remove the found tracks or not?
  
  AliHLTTPCHoughTransformer *fHoughTransformer; //!
  AliHLTTPCDigitRowData **fRowPointers; //!
  
  ClassDef(AliHLTTPCHoughEval,1) //Hough transform verfication class

};

#endif
