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
  AliAlgSens(const char* name = 0, int vid = 0, int iid = 0);
  virtual ~AliAlgSens();
  //
  virtual void addChild(AliAlgVol*);
  //
  void setDetector(AliAlgDet* det) { mDet = det; }
  AliAlgDet* getDetector() const { return mDet; }
  //
  int getSID() const { return mSID; }
  void setSID(int s) { mSID = s; }
  //
  void incrementStat() { mNProcPoints++; }
  //
  // derivatives calculation
  virtual void dPosTraDParCalib(const AliAlgPoint* pnt, double* deriv, int calibID, const AliAlgVol* parent = 0) const;
  virtual void dPosTraDParGeom(const AliAlgPoint* pnt, double* deriv, const AliAlgVol* parent = 0) const;
  //
  virtual void dPosTraDParGeomLOC(const AliAlgPoint* pnt, double* deriv) const;
  virtual void dPosTraDParGeomTRA(const AliAlgPoint* pnt, double* deriv) const;
  virtual void dPosTraDParGeomLOC(const AliAlgPoint* pnt, double* deriv, const AliAlgVol* parent) const;
  virtual void dPosTraDParGeomTRA(const AliAlgPoint* pnt, double* deriv, const AliAlgVol* parent) const;
  //
  void getModifiedMatrixT2LmodLOC(TGeoHMatrix& matMod, const double* delta) const;
  void getModifiedMatrixT2LmodTRA(TGeoHMatrix& matMod, const double* delta) const;
  //
  virtual void applyAlignmentFromMPSol();
  //
  void setAddError(double y, double z)
  {
    mAddError[0] = y;
    mAddError[1] = z;
  }
  const double* getAddError() const { return mAddError; }
  //
  virtual void prepareMatrixT2L();
  //
  virtual void setTrackingFrame();
  virtual bool isSensor() const { return true; }
  virtual void Print(const Option_t* opt = "") const;
  //
  virtual void updatePointByTrackInfo(AliAlgPoint* pnt, const trackParam_t* t) const;
  virtual void updateL2GRecoMatrices(const TClonesArray* algArr, const TGeoHMatrix* cumulDelta);
  //
  //  virtual AliAlgPoint* TrackPoint2AlgPoint(int pntId, const AliTrackPointArray* trpArr, const AliESDtrack* t) = 0; TODO(milettri): needs AliTrackPointArray AliESDtrack
  //
  virtual int finalizeStat(AliAlgDOFStat* h = 0);
  //
  virtual void prepareMatrixClAlg();
  virtual void prepareMatrixClAlgReco();
  const TGeoHMatrix& getMatrixClAlg() const { return mMatClAlg; }
  const TGeoHMatrix& getMatrixClAlgReco() const { return mMatClAlgReco; }
  void setMatrixClAlg(const TGeoHMatrix& m) { mMatClAlg = m; }
  void setMatrixClAlgReco(const TGeoHMatrix& m) { mMatClAlgReco = m; }
  //
 protected:
  //
  virtual bool IsSortable() const { return true; }
  virtual int Compare(const TObject* a) const;
  //
  // --------- dummies -----------
  AliAlgSens(const AliAlgSens&);
  AliAlgSens& operator=(const AliAlgSens&);
  //
 protected:
  //
  int mSID;                  // sensor id in detector
  double mAddError[2];       // additional error increment for measurement
  AliAlgDet* mDet;           // pointer on detector
  TGeoHMatrix mMatClAlg;     // reference cluster alignment matrix in tracking frame
  TGeoHMatrix mMatClAlgReco; // reco-time cluster alignment matrix in tracking frame

  //
  ClassDef(AliAlgSens, 1)
};
} // namespace align
} // namespace o2

#endif
