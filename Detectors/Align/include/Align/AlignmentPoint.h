// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include <TObject.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include "Align/AlignableSensor.h"
#include "ReconstructionDataFormats/Track.h"
#include "Framework/Logger.h"
#include "Align/utils.h"

namespace o2
{
namespace align
{

class AlignmentPoint : public TObject
{
 public:
  enum { kMaterialBit = BIT(14),        // point contains material
         kMeasurementBit = BIT(15),     // point contains measurement
         kUpdateFromTrackBit = BIT(16), // point needs to recalculate itself using track info
         kVaryELossBit = BIT(17),       // ELoss variation allowed
         kUseBzOnly = BIT(18),          // use only Bz component (ITS)
         kInvDir = BIT(19),             // propagation via this point is in decreasing X direction (upper cosmic leg)
         kStatOK = BIT(20)              // point is accounted in global statistics
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
  AlignmentPoint();
  ~AlignmentPoint() override = default;
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
  bool containsMaterial() const { return TestBit(kMaterialBit); }
  bool containsMeasurement() const { return TestBit(kMeasurementBit); }
  bool getNeedUpdateFromTrack() const { return TestBit(kUpdateFromTrackBit); }
  bool getELossVaried() const { return TestBit(kVaryELossBit); }
  bool getUseBzOnly() const { return TestBit(kUseBzOnly); }
  bool isInvDir() const { return TestBit(kInvDir); }
  bool isStatOK() const { return TestBit(kStatOK); }
  //
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
  void setELossVaried(bool v = true) { SetBit(kVaryELossBit, v); }
  void setContainsMaterial(bool v = true) { SetBit(kMaterialBit, v); }
  void setContainsMeasurement(bool v = true) { SetBit(kMeasurementBit, v); }
  void setNeedUpdateFromTrack(bool v = true) { SetBit(kUpdateFromTrackBit, v); }
  void setUseBzOnly(bool v = true) { SetBit(kUseBzOnly, v); }
  void setInvDir(bool v = true) { SetBit(kInvDir, v); }
  void setStatOK(bool v = true) { SetBit(kStatOK, v); }
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
  void Print(Option_t* option = "") const final;
  void Clear(Option_t* option = "") final;
  //
 protected:
  bool IsSortable() const final { return true; }
  int Compare(const TObject* a) const final;
  //
  // ---------- dummies ----------
  AlignmentPoint(const AlignmentPoint&);
  AlignmentPoint& operator=(const AlignmentPoint&);
  //
 protected:
  //
  int mMinLocVarID;         // The residuals/derivatives depend on fNLocExtPar params
                            // and point params>=mMinLocVarID.
  int mMaxLocVarID;         // The residuals/derivatives depend on fNLocExtPar params
                            // and point params<mMaxLocVarID.
                            // If the point contains materials, mMaxLocVarID also marks
                            // the parameters associated with this point
  char mDetID;              // DetectorID
  int16_t mSID;             // sensor ID in the detector
  float mAlphaSens;         // Alpha of tracking frame
  float mXSens;             // X of tracking frame
  float mCosDiagErr;        // Cos of Phi of rotation in YZ plane which diagonalize errors
  float mSinDiagErr;        // Sin of Phi of rotation in YZ plane which diagonalize errors
  float mErrDiag[2];        // diagonalized errors
  double mXYZTracking[3];   // X,Y,Z in tracking frame
  double mErrYZTracking[3]; // errors in tracking frame
  //
  float mX2X0;      // X2X0 seen by the track (including inclination)
  float mXTimesRho; // signed Density*Length seen by the track (including inclination)
  //
  int mNGloDOFs;                        // number of global DOFs this point depends on
  int mDGloOffs;                        // 1st entry slot of d/dGloPar in the AlgTrack fDResDGlo arrays
  float mMatCorrExp[kNMatDOFs];         // material correction due to ELoss expectation (non-diagonalized)
  float mMatCorrCov[kNMatDOFs];         // material correction delta covariance (diagonalized)
  float mMatDiag[kNMatDOFs][kNMatDOFs]; // matrix for  diagonalization of material effects errors
  //
  double mTrParamWSA[kNMatDOFs]; // workspace for tracks params at this point AFTER material correction
  double mTrParamWSB[kNMatDOFs]; // workspace for tracks params at this point BEFORE material correction

  AlignableSensor* mSensor; // sensor of this point

  ClassDef(AlignmentPoint, 1)
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
