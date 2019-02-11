#ifndef ALIESDPMDTRACK_H
#define ALIESDPMDTRACK_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

// Event Data Summary Class for pmd tracks
// This is part of the reconstructed ESD events
// for the PMD detector

#include "TObject.h"

class AliESDPmdTrack : public TObject {
 public:
  AliESDPmdTrack();
  virtual ~AliESDPmdTrack(){;}
  AliESDPmdTrack (const AliESDPmdTrack &PMDTrack);  // copy constructor
  AliESDPmdTrack &operator=(const AliESDPmdTrack &PMDTrack); // assignment op
  virtual void Copy(TObject &obj) const;

  void SetDetector(Int_t idet) {fDet = idet;}
  
  void SetClusterX(Float_t xglobal) {fX = xglobal;}
  void SetClusterY(Float_t yglobal) {fY = yglobal;}
  void SetClusterZ(Float_t zglobal) {fZ = zglobal;}
  void SetClusterADC(Float_t cluadc) {fCluADC = cluadc;}
  void SetClusterCells(Float_t ncell) {fNcell = (UChar_t)ncell;}
  void SetClusterPID(Float_t clupid) {fCluPID = clupid;}
  void SetSmn(Int_t smn) {fSmn = smn;}
  void SetClusterTrackNo(Int_t trno) {fTrackNo = trno;}
  void SetClusterTrackPid(Int_t trpid) {fTrackPid = trpid;}
  void SetClusterMatching(UShort_t mstat) {fClMatching = mstat;}
  void SetClusterSigmaX(Float_t sigx) {fSigX = sigx;}
  void SetClusterSigmaY(Float_t sigy) {fSigY = sigy;}

  Double_t GetClusterX() const {return fX;}
  Double_t GetClusterY() const {return fY;}
  Double_t GetClusterZ() const {return fZ;}
  Double_t GetClusterADC() const {return fCluADC;}
  Double_t GetClusterPID() const {return fCluPID;}
  UChar_t  GetClusterCells() const {return fNcell;}
  UChar_t  GetDetector() const {return fDet;}
  Int_t    GetSmn() const {return fSmn;}
  Int_t    GetClusterTrackNo() const {return fTrackNo;}
  Int_t    GetClusterTrackPid() const {return fTrackPid;}
  UShort_t GetClusterMatching() const {return fClMatching;}
  Double_t GetClusterSigmaX() const {return fSigX;}
  Double_t GetClusterSigmaY() const {return fSigY;}
  
  
 protected:

  Double32_t fX;      // Cluster X position
  Double32_t fY;      // Cluster Y position
  Double32_t fZ;      // Cluster Z position (vertex uncorrected)
  Double32_t fCluADC; // Cluster Energy in ADC
  Double32_t fCluPID; //[0.,1.,8] Cluster probability, 1: Photon, 0: Hadron
  UChar_t    fDet;    // Detector, 0:PRE, 1:CPV
  UChar_t    fNcell;  // Cluster cells
  Int_t      fSmn;    // Serial module number
  Int_t      fTrackNo; // Track number assigned to the clus from simulation
  Int_t      fTrackPid; // Track pid assigned to clus from simulation
  UShort_t   fClMatching; // Cluster of PRE matching with CPV
  Double32_t fSigX;       // Cluster x-width
  Double32_t fSigY;       // Cluster y-width

  ClassDef(AliESDPmdTrack,5)  //PMD ESD track class 

};

#endif 
