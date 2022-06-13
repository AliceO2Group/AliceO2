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

/// @file   AlignmentPoint.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Meausered point in the sensor.

/**
 * Meausered point in the sensor.
 * The measurement is in the tracking frame.
 * Apart from measurement may contain also material information.
 * Cashes residuals and track positions at its reference X
*/

#ifndef ALIGNMENTPOINT_H
#define ALIGNMENTPOINT_H

#include <Rtypes.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include "DetectorsCommonDataFormats/DetID.h"
#include "Align/AlignableSensor.h"
#include "ReconstructionDataFormats/Track.h"
#include "Framework/Logger.h"
#include "Align/utils.h"

namespace o2
{
namespace align
{

class AlignmentPoint
{
 public:
  using DetID = o2::detectors::DetID;

  enum BITS { kMaterialBit = 0x1 << 0,        // point contains material
              kMeasurementBit = 0x1 << 1,     // point contains measurement
              kUpdateFromTrackBit = 0x1 << 2, // point needs to recalculate itself using track info
              kVaryELossBit = 0x1 << 3,       // ELoss variation allowed
              kUseBzOnly = 0x1 << 4,          // use only Bz component (ITS)
              kInvDir = 0x1 << 5,             // propagation via this point is in decreasing X direction (upper cosmic leg)
              kStatOK = 0x1 << 6,             // point is accounted in global statistics

              kOptUMAT = 0x1 << 10,
              kOptDiag = 0x1 << 11,
              kOptWSA = 0x1 << 12,
              kOptWSB = 0x1 << 13

  };
  enum { kParY = 0 // track parameters
         ,
         kParZ,
         kParSnp,
         kParTgl,
         kParQ2Pt,
         kNMSPar = 4,
         kNELossPar = 1,
         kNMatDOFs = kNMSPar + kNELossPar };
  enum { kX,
         kY,
         kZ };
  //
  AlignmentPoint() = default;
  //
  void init();
  void updatePointByTrackInfo(const trackParam_t* t);
  //
  double getAlphaSens() const { return mAlphaSens; }
  double getXSens() const { return mXSens; }
  double getXPoint() const { return mXSens + getXTracking(); }
  double getXTracking() const { return mXYZTracking[0]; }
  double getYTracking() const { return mXYZTracking[1]; }
  double getZTracking() const { return mXYZTracking[2]; }
  const double* getYZTracking() const { return &mXYZTracking[1]; }
  const double* getXYZTracking() const { return mXYZTracking; }
  const double* getYZErrTracking() const { return mErrYZTracking; }

  const AlignableSensor* getSensor() const { return mSensor; }
  uint32_t getVolID() const { return mSensor->getVolID(); }
  void setSensor(AlignableSensor* s) { mSensor = s; }

  int getDetID() const { return mDetID; }
  int getSID() const { return mSID; }
  int getMinLocVarID() const { return mMinLocVarID; }
  int getMaxLocVarID() const { return mMaxLocVarID; }
  int getNMatPar() const;

  bool containsMaterial() const { return testBit(kMaterialBit); }
  bool containsMeasurement() const { return testBit(kMeasurementBit); }
  bool getNeedUpdateFromTrack() const { return testBit(kUpdateFromTrackBit); }
  bool getELossVaried() const { return testBit(kVaryELossBit); }
  bool getUseBzOnly() const { return testBit(kUseBzOnly); }
  bool isInvDir() const { return testBit(kInvDir); }
  bool isStatOK() const { return testBit(kStatOK); }
  //
  void setELossVaried(bool v = true) { setBit(kVaryELossBit, v); }
  void setContainsMaterial(bool v = true) { setBit(kMaterialBit, v); }
  void setContainsMeasurement(bool v = true) { setBit(kMeasurementBit, v); }
  void setNeedUpdateFromTrack(bool v = true) { setBit(kUpdateFromTrackBit, v); }
  void setUseBzOnly(bool v = true) { setBit(kUseBzOnly, v); }
  void setInvDir(bool v = true) { setBit(kInvDir, v); }
  void setStatOK(bool v = true) { setBit(kStatOK, v); }

  double getXTimesRho() const { return mXTimesRho; }
  double getX2X0() const { return mX2X0; }
  void setXTimesRho(double v) { mXTimesRho = v; }
  void setX2X0(double v) { mX2X0 = v; }
  //
  void setDetID(int id) { mDetID = (char)id; }
  void setSID(int id) { mSID = (int16_t)id; }
  //
  void setMinLocVarID(int id) { mMinLocVarID = id; }
  void setMaxLocVarID(int id) { mMaxLocVarID = id; }
  //
  void getResidualsDiag(const double* pos, double& resU, double& resV) const;
  void diagonalizeResiduals(double rY, double rZ, double& resU, double& resV) const;
  //
  void setAlphaSens(double a) { mAlphaSens = a; }
  void setXSens(double x) { mXSens = x; }
  void setXYZTracking(const double r[3])
  {
    for (int i = 3; i--;) {
      mXYZTracking[i] = r[i];
    }
  }
  void setXYZTracking(double x, double y, double z);
  void setYZErrTracking(double sy2, double syz, double sz2);
  void setYZErrTracking(const double* err)
  {
    for (int i = 3; i--;) {
      mErrYZTracking[i] = err[i];
    }
  }
  double getErrDiag(int i) const { return mErrDiag[i]; }
  //
  double* getTrParamWSA() const { return (double*)mTrParamWSA; }
  double* getTrParamWSB() const { return (double*)mTrParamWSB; }
  double getTrParamWSA(int ip) const { return mTrParamWSA[ip]; }
  double getTrParamWSB(int ip) const { return mTrParamWSB[ip]; }
  void getTrWSA(trackParam_t& etp) const;
  void getTrWSB(trackParam_t& etp) const;
  void setTrParamWSA(const double* param)
  {
    for (int i = 5; i--;) {
      mTrParamWSA[i] = param[i];
    }
  }
  void setTrParamWSB(const double* param)
  {
    for (int i = 5; i--;) {
      mTrParamWSB[i] = param[i];
    }
  }
  double getResidY() const { return getTrParamWSA(kParY) - getYTracking(); }
  double getResidZ() const { return getTrParamWSA(kParZ) - getZTracking(); }
  //
  void setMatCovDiagonalizationMatrix(const TMatrixD& d);
  void setMatCovDiag(const TVectorD& v);
  void setMatCovDiagElem(int i, double err2) { mMatCorrCov[i] = err2; }
  void unDiagMatCorr(const double* diag, double* nodiag) const;
  void diagMatCorr(const double* nodiag, double* diag) const;
  void unDiagMatCorr(const float* diag, float* nodiag) const;
  void diagMatCorr(const float* nodiag, float* diag) const;
  //
  void setMatCorrExp(double* p)
  {
    for (int i = 5; i--;) {
      mMatCorrExp[i] = p[i];
    }
  }
  float* getMatCorrExp() const { return (float*)mMatCorrExp; }
  float* getMatCorrCov() const { return (float*)mMatCorrCov; }
  //
  void getXYZGlo(double r[3]) const;
  double getPhiGlo() const;
  int getAliceSector() const;
  //
  int getNGloDOFs() const { return mNGloDOFs; }
  int getDGloOffs() const { return mDGloOffs; }
  void setNGloDOFs(int n) { mNGloDOFs = n; }
  void setDGloOffs(int n) { mDGloOffs = n; }
  //
  void incrementStat();
  //
  virtual void dumpCoordinates() const;
  void print(uint16_t opt) const;
  void clear();
  //
  bool isAfter(const AlignmentPoint& pnt) const;

 protected:
  //
  // ---------- dummies ----------
  void setBit(BITS b, bool v)
  {
    if (v) {
      mBits |= b;
    } else {
      mBits &= ~(b & 0xffff);
    }
  }
  bool testBit(BITS b) const
  {
    return mBits & b;
  }

 protected:
  //
  int mMinLocVarID = -1;          // The residuals/derivatives depend on fNLocExtPar params
                                  // and point params>=mMinLocVarID.
  int mMaxLocVarID = -1;          // The residuals/derivatives depend on fNLocExtPar params
                                  // and point params<mMaxLocVarID.
                                  // If the point contains materials, mMaxLocVarID also marks
                                  // the parameters associated with this point
  DetID mDetID{};                 // DetectorID
  int16_t mSID = -1;              // sensor ID in the detector
  uint16_t mBits = 0;             // flags
  float mAlphaSens = 0;           // Alpha of tracking frame
  float mXSens = 0;               // X of tracking frame
  float mCosDiagErr = 0;          // Cos of Phi of rotation in YZ plane which diagonalize errors
  float mSinDiagErr = 0;          // Sin of Phi of rotation in YZ plane which diagonalize errors
  float mErrDiag[2] = {0};        // diagonalized errors
  double mXYZTracking[3] = {0};   // X,Y,Z in tracking frame
  double mErrYZTracking[3] = {0}; // errors in tracking frame
  //
  float mX2X0 = 0;      // X2X0 seen by the track (including inclination)
  float mXTimesRho = 0; // signed Density*Length seen by the track (including inclination)
  //
  int mNGloDOFs = 0;                         // number of global DOFs this point depends on
  int mDGloOffs = 0;                         // 1st entry slot of d/dGloPar in the AlgTrack fDResDGlo arrays
  float mMatCorrExp[kNMatDOFs] = {};         // material correction due to ELoss expectation (non-diagonalized)
  float mMatCorrCov[kNMatDOFs] = {};         // material correction delta covariance (diagonalized)
  float mMatDiag[kNMatDOFs][kNMatDOFs] = {}; // matrix for  diagonalization of material effects errors
  //
  double mTrParamWSA[kNMatDOFs] = {}; // workspace for tracks params at this point AFTER material correction
  double mTrParamWSB[kNMatDOFs] = {}; // workspace for tracks params at this point BEFORE material correction

  AlignableSensor* mSensor = nullptr; // sensor of this point

  ClassDefNV(AlignmentPoint, 1)
};

//____________________________________________________
inline void AlignmentPoint::setXYZTracking(double x, double y, double z)
{
  // assign tracking coordinates
  mXYZTracking[0] = x;
  mXYZTracking[1] = y;
  mXYZTracking[2] = z;
}

//____________________________________________________
inline void AlignmentPoint::setYZErrTracking(double sy2, double syz, double sz2)
{
  // assign tracking coordinates
  mErrYZTracking[0] = sy2;
  mErrYZTracking[1] = syz;
  mErrYZTracking[2] = sz2;
}

inline int AlignmentPoint::getNMatPar() const
{
  // get number of free params for material descriptoin
  return containsMaterial() ? (getELossVaried() ? kNMSPar + kNELossPar : kNMSPar) : 0;
}

//_____________________________________
inline void AlignmentPoint::diagonalizeResiduals(double rY, double rZ, double& resU, double& resV) const
{
  // rotate residuals to frame where their error matrix is diagonal
  resU = mCosDiagErr * rY - mSinDiagErr * rZ;
  resV = mSinDiagErr * rY + mCosDiagErr * rZ;
  //
}

//_____________________________________
inline void AlignmentPoint::getResidualsDiag(const double* pos, double& resU, double& resV) const
{
  // calculate residuals in the frame where the errors are diagonal, given the position
  // of the track in the standard tracking frame
  diagonalizeResiduals(pos[0] - mXYZTracking[1], pos[1] - mXYZTracking[2], resU, resV);
  //
}

//__________________________________________________________________
inline void AlignmentPoint::incrementStat()
{
  // increment statistics for detectors this point depends on
  mSensor->incrementStat();
  setStatOK();
}
} // namespace align
} // namespace o2
#endif
