// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgPoint.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Meausered point in the sensor.

/**
 * Meausered point in the sensor.
 * The measurement is in the tracking frame.
 * Apart from measurement may contain also material information.
 * Cashes residuals and track positions at its reference X
*/

#ifndef ALIALGPOINT_H
#define ALIALGPOINT_H

#include <TObject.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include "Align/AliAlgSens.h"
#include "ReconstructionDataFormats/Track.h"
#include "Framework/Logger.h"
#include "Align/AliAlgAux.h"

namespace o2
{
namespace align
{

class AliAlgPoint : public TObject
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
  AliAlgPoint();
  virtual ~AliAlgPoint() {}
  //
  void Init();
  void UpdatePointByTrackInfo(const trackParam_t* t);
  //
  double GetAlphaSens() const { return fAlphaSens; }
  double GetXSens() const { return fXSens; }
  double GetXPoint() const { return fXSens + GetXTracking(); }
  double GetXTracking() const { return fXYZTracking[0]; }
  double GetYTracking() const { return fXYZTracking[1]; }
  double GetZTracking() const { return fXYZTracking[2]; }
  const double* GetYZTracking() const { return &fXYZTracking[1]; }
  const double* GetXYZTracking() const { return fXYZTracking; }
  const double* GetYZErrTracking() const { return fErrYZTracking; }

  const AliAlgSens* GetSensor() const { return fSensor; }
  uint32_t GetVolID() const { return fSensor->GetVolID(); }
  void SetSensor(AliAlgSens* s) { fSensor = s; }

  int GetDetID() const { return fDetID; }
  int GetSID() const { return fSID; }
  int GetMinLocVarID() const { return fMinLocVarID; }
  int GetMaxLocVarID() const { return fMaxLocVarID; }
  int GetNMatPar() const;
  bool ContainsMaterial() const { return TestBit(kMaterialBit); }
  bool ContainsMeasurement() const { return TestBit(kMeasurementBit); }
  bool GetNeedUpdateFromTrack() const { return TestBit(kUpdateFromTrackBit); }
  bool GetELossVaried() const { return TestBit(kVaryELossBit); }
  bool GetUseBzOnly() const { return TestBit(kUseBzOnly); }
  bool IsInvDir() const { return TestBit(kInvDir); }
  bool IsStatOK() const { return TestBit(kStatOK); }
  //
  double GetXTimesRho() const { return fXTimesRho; }
  double GetX2X0() const { return fX2X0; }
  void SetXTimesRho(double v) { fXTimesRho = v; }
  void SetX2X0(double v) { fX2X0 = v; }
  //
  void SetDetID(int id) { fDetID = (char)id; }
  void SetSID(int id) { fSID = (int16_t)id; }
  //
  void SetMinLocVarID(int id) { fMinLocVarID = id; }
  void SetMaxLocVarID(int id) { fMaxLocVarID = id; }
  void SetELossVaried(bool v = true) { SetBit(kVaryELossBit, v); }
  void SetContainsMaterial(bool v = true) { SetBit(kMaterialBit, v); }
  void SetContainsMeasurement(bool v = true) { SetBit(kMeasurementBit, v); }
  void SetNeedUpdateFromTrack(bool v = true) { SetBit(kUpdateFromTrackBit, v); }
  void SetUseBzOnly(bool v = true) { SetBit(kUseBzOnly, v); }
  void SetInvDir(bool v = true) { SetBit(kInvDir, v); }
  void SetStatOK(bool v = true) { SetBit(kStatOK, v); }
  //
  void GetResidualsDiag(const double* pos, double& resU, double& resV) const;
  void DiagonalizeResiduals(double rY, double rZ, double& resU, double& resV) const;
  //
  void SetAlphaSens(double a) { fAlphaSens = a; }
  void SetXSens(double x) { fXSens = x; }
  void SetXYZTracking(const double r[3])
  {
    for (int i = 3; i--;)
      fXYZTracking[i] = r[i];
  }
  void SetXYZTracking(double x, double y, double z);
  void SetYZErrTracking(double sy2, double syz, double sz2);
  void SetYZErrTracking(const double* err)
  {
    for (int i = 3; i--;)
      fErrYZTracking[i] = err[i];
  }
  double GetErrDiag(int i) const { return fErrDiag[i]; }
  //
  double* GetTrParamWSA() const { return (double*)fTrParamWSA; }
  double* GetTrParamWSB() const { return (double*)fTrParamWSB; }
  double GetTrParamWSA(int ip) const { return fTrParamWSA[ip]; }
  double GetTrParamWSB(int ip) const { return fTrParamWSB[ip]; }
  void GetTrWSA(trackParam_t& etp) const;
  void GetTrWSB(trackParam_t& etp) const;
  void SetTrParamWSA(const double* param)
  {
    for (int i = 5; i--;)
      fTrParamWSA[i] = param[i];
  }
  void SetTrParamWSB(const double* param)
  {
    for (int i = 5; i--;)
      fTrParamWSB[i] = param[i];
  }
  double GetResidY() const { return GetTrParamWSA(kParY) - GetYTracking(); }
  double GetResidZ() const { return GetTrParamWSA(kParZ) - GetZTracking(); }
  //
  void SetMatCovDiagonalizationMatrix(const TMatrixD& d);
  void SetMatCovDiag(const TVectorD& v);
  void SetMatCovDiagElem(int i, double err2) { fMatCorrCov[i] = err2; }
  void UnDiagMatCorr(const double* diag, double* nodiag) const;
  void DiagMatCorr(const double* nodiag, double* diag) const;
  void UnDiagMatCorr(const float* diag, float* nodiag) const;
  void DiagMatCorr(const float* nodiag, float* diag) const;
  //
  void SetMatCorrExp(double* p)
  {
    for (int i = 5; i--;)
      fMatCorrExp[i] = p[i];
  }
  float* GetMatCorrExp() const { return (float*)fMatCorrExp; }
  float* GetMatCorrCov() const { return (float*)fMatCorrCov; }
  //
  void GetXYZGlo(double r[3]) const;
  double GetPhiGlo() const;
  int GetAliceSector() const;
  //
  int GetNGloDOFs() const { return fNGloDOFs; }
  int GetDGloOffs() const { return fDGloOffs; }
  void SetNGloDOFs(int n) { fNGloDOFs = n; }
  void SetDGloOffs(int n) { fDGloOffs = n; }
  //
  void IncrementStat();
  //
  virtual void DumpCoordinates() const;
  virtual void Print(Option_t* option = "") const;
  virtual void Clear(Option_t* option = "");
  //
 protected:
  virtual bool IsSortable() const { return true; }
  virtual int Compare(const TObject* a) const;
  //
  // ---------- dummies ----------
  AliAlgPoint(const AliAlgPoint&);
  AliAlgPoint& operator=(const AliAlgPoint&);
  //
 protected:
  //
  int fMinLocVarID;         // The residuals/derivatives depend on fNLocExtPar params
                            // and point params>=fMinLocVarID.
  int fMaxLocVarID;         // The residuals/derivatives depend on fNLocExtPar params
                            // and point params<fMaxLocVarID.
                            // If the point contains materials, fMaxLocVarID also marks
                            // the parameters associated with this point
  char fDetID;              // DetectorID
  int16_t fSID;             // sensor ID in the detector
  float fAlphaSens;         // Alpha of tracking frame
  float fXSens;             // X of tracking frame
  float fCosDiagErr;        // Cos of Phi of rotation in YZ plane which diagonalize errors
  float fSinDiagErr;        // Sin of Phi of rotation in YZ plane which diagonalize errors
  float fErrDiag[2];        // diagonalized errors
  double fXYZTracking[3];   // X,Y,Z in tracking frame
  double fErrYZTracking[3]; // errors in tracking frame
  //
  float fX2X0;      // X2X0 seen by the track (including inclination)
  float fXTimesRho; // signed Density*Length seen by the track (including inclination)
  //
  int fNGloDOFs;                        // number of global DOFs this point depends on
  int fDGloOffs;                        // 1st entry slot of d/dGloPar in the AlgTrack fDResDGlo arrays
  float fMatCorrExp[kNMatDOFs];         // material correction due to ELoss expectation (non-diagonalized)
  float fMatCorrCov[kNMatDOFs];         // material correction delta covariance (diagonalized)
  float fMatDiag[kNMatDOFs][kNMatDOFs]; // matrix for  diagonalization of material effects errors
  //
  double fTrParamWSA[kNMatDOFs]; // workspace for tracks params at this point AFTER material correction
  double fTrParamWSB[kNMatDOFs]; // workspace for tracks params at this point BEFORE material correction

  AliAlgSens* fSensor; // sensor of this point

  ClassDef(AliAlgPoint, 1)
};

//____________________________________________________
inline void AliAlgPoint::SetXYZTracking(double x, double y, double z)
{
  // assign tracking coordinates
  fXYZTracking[0] = x;
  fXYZTracking[1] = y;
  fXYZTracking[2] = z;
}

//____________________________________________________
inline void AliAlgPoint::SetYZErrTracking(double sy2, double syz, double sz2)
{
  // assign tracking coordinates
  fErrYZTracking[0] = sy2;
  fErrYZTracking[1] = syz;
  fErrYZTracking[2] = sz2;
}

inline int AliAlgPoint::GetNMatPar() const
{
  // get number of free params for material descriptoin
  return ContainsMaterial() ? (GetELossVaried() ? kNMSPar + kNELossPar : kNMSPar) : 0;
}

//_____________________________________
inline void AliAlgPoint::DiagonalizeResiduals(double rY, double rZ, double& resU, double& resV) const
{
  // rotate residuals to frame where their error matrix is diagonal
  resU = fCosDiagErr * rY - fSinDiagErr * rZ;
  resV = fSinDiagErr * rY + fCosDiagErr * rZ;
  //
}

//_____________________________________
inline void AliAlgPoint::GetResidualsDiag(const double* pos, double& resU, double& resV) const
{
  // calculate residuals in the frame where the errors are diagonal, given the position
  // of the track in the standard tracking frame
  DiagonalizeResiduals(pos[0] - fXYZTracking[1], pos[1] - fXYZTracking[2], resU, resV);
  //
}

//__________________________________________________________________
inline void AliAlgPoint::IncrementStat()
{
  // increment statistics for detectors this point depends on
  fSensor->IncrementStat();
  SetStatOK();
}
} // namespace align
} // namespace o2
#endif
