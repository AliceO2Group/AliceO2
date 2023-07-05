// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableSensor.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  End-chain alignment volume in detector branch, where the actual measurement is done.

#ifndef ALIGNABLESENSOR_H
#define ALIGNABLESENSOR_H

#include <TMath.h>
#include <TObjArray.h>
#include "ReconstructionDataFormats/BaseCluster.h"
#include "Align/AlignableVolume.h"
#include "Align/utils.h"
#include <vector>

namespace o2
{
namespace align
{

class AlignableDetector;
class AlignmentPoint;

class AlignableSensor : public AlignableVolume
{
 public:
  //
  using ClusterF = o2::BaseCluster<float>;

  AlignableSensor() = default;
  AlignableSensor(const char* name, int vid, int iid, Controller* ctr);
  ~AlignableSensor() override = default;
  //
  void addChild(AlignableVolume*) override;
  //
  void setDetector(AlignableDetector* det) { mDet = det; }
  AlignableDetector* getDetector() const { return mDet; }
  //
  int getSID() const { return mSID; }
  void setSID(int s) { mSID = s; }
  //
  void incrementStat() { mNProcPoints++; }
  //
  // derivatives calculation
  virtual void dPosTraDParCalib(const AlignmentPoint* pnt, double* deriv, int calibID, const AlignableVolume* parent = nullptr) const;
  virtual void dPosTraDParGeom(const AlignmentPoint* pnt, double* deriv, const AlignableVolume* parent = nullptr) const;
  //
  virtual void dPosTraDParGeomLOC(const AlignmentPoint* pnt, double* deriv) const;
  virtual void dPosTraDParGeomTRA(const AlignmentPoint* pnt, double* deriv) const;
  virtual void dPosTraDParGeomLOC(const AlignmentPoint* pnt, double* deriv, const AlignableVolume* parent) const;
  virtual void dPosTraDParGeomTRA(const AlignmentPoint* pnt, double* deriv, const AlignableVolume* parent) const;
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
  bool isSensor() const override { return true; }
  void Print(const Option_t* opt = "") const override;
  //
  virtual void updatePointByTrackInfo(AlignmentPoint* pnt, const trackParam_t* t) const;
  void updateL2GRecoMatrices(const std::vector<o2::detectors::AlignParam>& algArr, const TGeoHMatrix* cumulDelta) override;
  //
  int finalizeStat() override;
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
  bool IsSortable() const override { return true; }
  int Compare(const TObject* a) const override;
  //
  // --------- dummies -----------
  AlignableSensor(const AlignableSensor&);
  AlignableSensor& operator=(const AlignableSensor&);
  //
 protected:
  //
  int mSID = -1;                     // sensor id in detector
  double mAddError[2] = {0, 0};      // additional error increment for measurement
  AlignableDetector* mDet = nullptr; // pointer on detector
  TGeoHMatrix mMatClAlg;             // reference cluster alignment matrix in tracking frame, i.e. the matrix we want to correct
  TGeoHMatrix mMatClAlgReco;         // reco-time cluster alignment matrix in tracking frame

  //
  ClassDefOverride(AlignableSensor, 1)
};
} // namespace align
} // namespace o2

#endif
