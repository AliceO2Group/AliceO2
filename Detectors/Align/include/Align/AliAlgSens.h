// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgSens.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  End-chain alignment volume in detector branch, where the actual measurement is done.

#ifndef ALIALGSENS_H
#define ALIALGSENS_H

#include <TMath.h>
#include <TObjArray.h>

#include "Align/AliAlgVol.h"
#include "Align/AliAlgDOFStat.h"
#include "Align/AliAlgAux.h"

//class AliTrackPointArray;
//class AliESDtrack;
class TCloneArray;

namespace o2
{
namespace align
{

class AliAlgDet;
class AliAlgPoint;

class AliAlgSens : public AliAlgVol
{
 public:
  //
  AliAlgSens(const char* name = 0, Int_t vid = 0, Int_t iid = 0);
  virtual ~AliAlgSens();
  //
  virtual void AddChild(AliAlgVol*);
  //
  void SetDetector(AliAlgDet* det) { fDet = det; }
  AliAlgDet* GetDetector() const { return fDet; }
  //
  Int_t GetSID() const { return fSID; }
  void SetSID(int s) { fSID = s; }
  //
  void IncrementStat() { fNProcPoints++; }
  //
  // derivatives calculation
  virtual void DPosTraDParCalib(const AliAlgPoint* pnt, double* deriv, int calibID, const AliAlgVol* parent = 0) const;
  virtual void DPosTraDParGeom(const AliAlgPoint* pnt, double* deriv, const AliAlgVol* parent = 0) const;
  //
  virtual void DPosTraDParGeomLOC(const AliAlgPoint* pnt, double* deriv) const;
  virtual void DPosTraDParGeomTRA(const AliAlgPoint* pnt, double* deriv) const;
  virtual void DPosTraDParGeomLOC(const AliAlgPoint* pnt, double* deriv, const AliAlgVol* parent) const;
  virtual void DPosTraDParGeomTRA(const AliAlgPoint* pnt, double* deriv, const AliAlgVol* parent) const;
  //
  void GetModifiedMatrixT2LmodLOC(TGeoHMatrix& matMod, const Double_t* delta) const;
  void GetModifiedMatrixT2LmodTRA(TGeoHMatrix& matMod, const Double_t* delta) const;
  //
  virtual void ApplyAlignmentFromMPSol();
  //
  void SetAddError(double y, double z)
  {
    fAddError[0] = y;
    fAddError[1] = z;
  }
  const Double_t* GetAddError() const { return fAddError; }
  //
  virtual void PrepareMatrixT2L();
  //
  virtual void SetTrackingFrame();
  virtual Bool_t IsSensor() const { return kTRUE; }
  virtual void Print(const Option_t* opt = "") const;
  //
  virtual void UpdatePointByTrackInfo(AliAlgPoint* pnt, const trackParam_t* t) const;
  virtual void UpdateL2GRecoMatrices(const TClonesArray* algArr, const TGeoHMatrix* cumulDelta);
  //
  //  virtual AliAlgPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t) = 0; TODO(milettri): needs AliTrackPointArray AliESDtrack
  //
  virtual Int_t FinalizeStat(AliAlgDOFStat* h = 0);
  //
  virtual void PrepareMatrixClAlg();
  virtual void PrepareMatrixClAlgReco();
  const TGeoHMatrix& GetMatrixClAlg() const { return fMatClAlg; }
  const TGeoHMatrix& GetMatrixClAlgReco() const { return fMatClAlgReco; }
  void SetMatrixClAlg(const TGeoHMatrix& m) { fMatClAlg = m; }
  void SetMatrixClAlgReco(const TGeoHMatrix& m) { fMatClAlgReco = m; }
  //
 protected:
  //
  virtual Bool_t IsSortable() const { return kTRUE; }
  virtual Int_t Compare(const TObject* a) const;
  //
  // --------- dummies -----------
  AliAlgSens(const AliAlgSens&);
  AliAlgSens& operator=(const AliAlgSens&);
  //
 protected:
  //
  Int_t fSID;                // sensor id in detector
  Double_t fAddError[2];     // additional error increment for measurement
  AliAlgDet* fDet;           // pointer on detector
  TGeoHMatrix fMatClAlg;     // reference cluster alignment matrix in tracking frame
  TGeoHMatrix fMatClAlgReco; // reco-time cluster alignment matrix in tracking frame

  //
  ClassDef(AliAlgSens, 1)
};
} // namespace align
} // namespace o2

#endif
