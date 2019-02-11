#ifndef ALIESDHLTTRACK_H
#define ALIESDHLTTRACK_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */
//-------------------------------------------------------------------------
//                          Class AliESDHLTtrack
//   This is the class to handle HLT reconstruted TPC tracks
//-------------------------------------------------------------------------
#include "TObject.h"
#include "TMath.h"

class AliKalmanTrack;

class AliESDHLTtrack : public TObject {
public:
  AliESDHLTtrack();
  virtual ~AliESDHLTtrack() {}

  // getters
  Int_t GetNHits() const {return fNHits;}

  Int_t GetMCid() const {return fMCid;}

  Int_t GetWeight() const {return fWeight;}

  Bool_t ComesFromMainVertex() const {return fFromMainVertex;}

  Int_t GetFirstRow() const {return fRowRange[0];}
  Int_t GetLastRow()  const {return fRowRange[1];}
  Int_t GetSector()   const {return fSector;}

  Double_t GetFirstPointX() const {return fFirstPoint[0];}
  Double_t GetFirstPointY() const {return fFirstPoint[1];}
  Double_t GetFirstPointZ() const {return fFirstPoint[2];}
  Double_t GetLastPointX() const {return fLastPoint[0];}
  Double_t GetLastPointY() const {return fLastPoint[1];}
  Double_t GetLastPointZ() const {return fLastPoint[2];}

  Int_t GetCharge() const {return fQ;}
  Double_t GetPt() const {return fPt;}
  Double_t GetTgl() const {return fTanl;}
  Double_t GetPsi() const {return fPsi;}

  Double_t GetPterr()  const {return fPterr;}
  Double_t GetPsierr() const {return fPsierr;}
  Double_t GetTglerr() const {return fTanlerr;}

  Float_t    GetBinX()   const {return fBinX;}
  Float_t    GetBinY()   const {return fBinY;}
  Float_t    GetSizeX()  const {return fSizeX;}
  Float_t    GetSizeY()  const {return fSizeY;}

  Double_t GetPx() const {return fPt*TMath::Cos(fPsi);}
  Double_t GetPy() const {return fPt*TMath::Sin(fPsi);}
  Double_t GetPz() const {return fPt*fTanl;}

  Double_t GetP() const;
  Double_t GetPseudoRapidity() const;

  Float_t GetPID() const {return fPID;}

  // setters
  void SetNHits(Int_t f) {fNHits = f;}

  void SetMCid(Int_t f) {fMCid = f;}

  void SetWeight(Int_t f) {fWeight = f;}
  
  void ComesFromMainVertex(Bool_t f) {fFromMainVertex = f;}
  
  void SetRowRange(Int_t f,Int_t g) {fRowRange[0]=f; fRowRange[1]=g;}
  void SetSector(Int_t f) {fSector = f;}

  void SetFirstPoint(Double_t f,Double_t g,Double_t h) {fFirstPoint[0]=f; fFirstPoint[1]=g; fFirstPoint[2]=h;}
  void SetLastPoint(Double_t f,Double_t g,Double_t h) {fLastPoint[0]=f; fLastPoint[1]=g; fLastPoint[2]=h;}

  void SetCharge(Int_t f) {fQ = f;}
  void SetTgl(Double_t f) {fTanl =f;}
  void SetPsi(Double_t f) {fPsi = f;}
  void SetPt(Double_t f) {fPt = f;}

  void SetPterr(Double_t f) {fPterr = f;}
  void SetPsierr(Double_t f) {fPsierr = f;}
  void SetTglerr(Double_t f) {fTanlerr = f;}

  void SetBinXY(Float_t binx,Float_t biny,Float_t sizex,Float_t sizey) {fBinX = binx; fBinY = biny; fSizeX = sizex; fSizeY = sizey;}

  void SetPID(Float_t pid) {fPID = pid;}

  Bool_t UpdateTrackParams(const AliKalmanTrack *t);

protected:
  UShort_t fNHits;  // Number of assigned clusters

  Int_t fMCid;  //Assigned id from MC data.

  UShort_t fWeight; //Weight associated to Hough Transform

  Bool_t   fFromMainVertex; // true if tracks origin is the main vertex, otherwise false
  
  Int_t fRowRange[2]; //Subsector where this track was build
  UShort_t fSector;      //Sector # where  this track was build

  Float_t fFirstPoint[3]; //First track point in TPC
  Float_t fLastPoint[3];  //Last track point in TPC

  Int_t    fQ;    //track charge
  Float_t fTanl; //tan of dipangle
  Float_t fPsi;  //azimuthal angle of the momentum 
  Float_t fPt;   //transverse momentum

  Float_t fPterr;   //Pt error
  Float_t fPsierr;  //Psi error
  Float_t fTanlerr; //Error of Tangent lambda

  Float_t fBinX;  //X bin?
  Float_t fBinY;  //Y bin?
  Float_t fSizeX; //X size?
  Float_t fSizeY; //Y size?
  
  Float_t fPID; //so far filled only for conformal mapper tracks

  ClassDef(AliESDHLTtrack,3) //ESD HLT track class
};

#endif
