// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgDet.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Base class for detector: wrapper for set of volumes

#ifndef ALIALGDET_H
#define ALIALGDET_H

#include <TNamed.h>
#include <TObjArray.h>
#include <stdio.h>
#include "Align/AliAlgAux.h"
#include "Align/AliAlgTrack.h"
#include "Align/AliAlgDOFStat.h"
#include "Align/AliAlgPoint.h"
#include "Align/AliAlgSens.h"
#include "Align/AliAlgVol.h"
#include "Align/AliAlgSteer.h"
//#include "AliESDtrack.h"
//class AliTrackPointArray;

class TH1;

namespace o2
{
namespace align
{

//TODO(milettri) : fix possibly incompatible Detector IDs of O2 and AliROOT
class AliAlgDet : public TNamed
{
 public:
  enum { kInitGeomDone = BIT(14),
         kInitDOFsDone = BIT(15) };
  enum { kNMaxKalibDOF = 64 };
  //
  AliAlgDet();
  AliAlgDet(const char* name, const char* title = "", int id = -1) : TNamed(name, title) { SetUniqueID(id); };
  virtual ~AliAlgDet();
  int GetDetID() const { return GetUniqueID(); }
  detectors::DetID GetO2DetID() const { return detectors::DetID(GetUniqueID()); };
  void SetDetID(uint32_t tp);
  void SetDetID(detectors::DetID id) { SetUniqueID(id); }
  //
  virtual void CacheReferenceOCDB();
  virtual void AcknowledgeNewRun(int run);
  virtual void UpdateL2GRecoMatrices();
  virtual void ApplyAlignmentFromMPSol();
  //
  int VolID2SID(int vid) const;
  int SID2VolID(int sid) const { return sid < GetNSensors() ? fSID2VolID[sid] : -1; } //todo
  int GetNSensors() const { return fSensors.GetEntriesFast(); }
  int GetNVolumes() const { return fVolumes.GetEntriesFast(); }
  int GetVolIDMin() const { return fVolIDMin; }
  int GetVolIDMax() const { return fVolIDMax; }
  bool SensorOfDetector(int vid) const { return vid >= fVolIDMin && vid <= fVolIDMax; }
  void SetAddError(double y, double z);
  const double* GetAddError() const { return fAddError; }
  //
  int GetNPoints() const { return fNPoints; }
  //
  void SetAlgSteer(AliAlgSteer* s) { fAlgSteer = s; }
  AliAlgSens* GetSensor(int id) const { return (AliAlgSens*)fSensors.UncheckedAt(id); }
  AliAlgSens* GetSensorByVolId(int vid) const
  {
    int sid = VolID2SID(vid);
    return sid < 0 ? 0 : GetSensor(sid);
  }
  AliAlgSens* GetSensor(const char* symname) const { return (AliAlgSens*)fSensors.FindObject(symname); }
  AliAlgVol* GetVolume(int id) const { return (AliAlgVol*)fVolumes.UncheckedAt(id); }
  AliAlgVol* GetVolume(const char* symname) const { return (AliAlgVol*)fVolumes.FindObject(symname); }
  //
  bool OwnsDOFID(int id) const;
  AliAlgVol* GetVolOfDOFID(int id) const;
  //
  int GetDetLabel() const { return (GetDetID() + 1) * 1000000; }
  void SetFreeDOF(int dof);
  void FixDOF(int dof);
  void SetFreeDOFPattern(uint64_t pat)
  {
    fCalibDOF = pat;
    CalcFree();
  }
  bool IsFreeDOF(int dof) const { return (fCalibDOF & (0x1 << dof)) != 0; }
  bool IsCondDOF(int dof) const;
  uint64_t GetFreeDOFPattern() const { return fCalibDOF; }
  int GetNProcessedPoints() const { return fNProcPoints; }
  virtual const char* GetCalibDOFName(int) const { return 0; }
  virtual double GetCalibDOFVal(int) const { return 0; }
  virtual double GetCalibDOFValWithCal(int) const { return 0; }
  //
  virtual int InitGeom();
  virtual int AssignDOFs();
  virtual void InitDOFs();
  virtual void Terminate();
  void FillDOFStat(AliAlgDOFStat* dofst = 0) const;
  virtual void AddVolume(AliAlgVol* vol);
  virtual void DefineVolumes();
  virtual void DefineMatrices();
  virtual void Print(const Option_t* opt = "") const;
  //  virtual int ProcessPoints(const AliESDtrack* esdTr, AliAlgTrack* algTrack, bool inv = false); FIXME(milettri): needs AliESDtrack
  virtual void UpdatePointByTrackInfo(AliAlgPoint* pnt, const trackParam_t* t) const;
  virtual void SetUseErrorParam(int v = 0);
  int GetUseErrorParam() const { return fUseErrorParam; }
  //
  //  virtual bool AcceptTrack(const AliESDtrack* trc, int trtype) const = 0; FIXME(milettri): needs AliESDtrack
  //  bool CheckFlags(const AliESDtrack* trc, int trtype) const; FIXME(milettri): needs AliESDtrack
  //
  virtual AliAlgPoint* GetPointFromPool();
  virtual void ResetPool();
  virtual void WriteSensorPositions(const char* outFName);
  //
  void SetInitGeomDone() { SetBit(kInitGeomDone); }
  bool GetInitGeomDone() const { return TestBit(kInitGeomDone); }
  //
  void SetInitDOFsDone() { SetBit(kInitDOFsDone); }
  bool GetInitDOFsDone() const { return TestBit(kInitDOFsDone); }
  void FixNonSensors();
  void SetFreeDOFPattern(uint32_t pat = 0xffffffff, int lev = -1, const char* match = 0);
  void SetDOFCondition(int dof, float condErr, int lev = -1, const char* match = 0);
  int SelectVolumes(TObjArray* arr, int lev = -1, const char* match = 0);
  //
  int GetNDOFs() const { return fNDOFs; }
  int GetNCalibDOFs() const { return fNCalibDOF; }
  int GetNCalibDOFsFree() const { return fNCalibDOFFree; }
  //
  void SetDisabled(int tp, bool v)
  {
    fDisabled[tp] = v;
    SetObligatory(tp, !v);
  }
  void SetDisabled()
  {
    SetDisabledColl();
    SetDisabledCosm();
  }
  void SetDisabledColl(bool v = true) { SetDisabled(AliAlgAux::kColl, v); }
  void SetDisabledCosm(bool v = true) { SetDisabled(AliAlgAux::kCosm, v); }
  bool IsDisabled(int tp) const { return fDisabled[tp]; }
  bool IsDisabled() const { return IsDisabledColl() && IsDisabledCosm(); }
  bool IsDisabledColl() const { return IsDisabled(AliAlgAux::kColl); }
  bool IsDisabledCosm() const { return IsDisabled(AliAlgAux::kCosm); }
  //
  void SetTrackFlagSel(int tp, uint64_t f) { fTrackFlagSel[tp] = f; }
  void SetTrackFlagSelColl(uint64_t f) { SetTrackFlagSel(AliAlgAux::kColl, f); }
  void SetTrackFlagSelCosm(uint64_t f) { SetTrackFlagSel(AliAlgAux::kCosm, f); }
  uint64_t GetTrackFlagSel(int tp) const { return fTrackFlagSel[tp]; }
  uint64_t GetTrackFlagSelColl() const { return GetTrackFlagSel(AliAlgAux::kColl); }
  uint64_t GetTrackFlagSelCosm() const { return GetTrackFlagSel(AliAlgAux::kCosm); }
  //
  void SetNPointsSel(int tp, int n) { fNPointsSel[tp] = n; }
  void SetNPointsSelColl(int n) { SetNPointsSel(AliAlgAux::kColl, n); }
  void SetNPointsSelCosm(int n) { SetNPointsSel(AliAlgAux::kCosm, n); }
  int GetNPointsSel(int tp) const { return fNPointsSel[tp]; }
  int GetNPointsSelColl() const { return GetNPointsSel(AliAlgAux::kColl); }
  int GetNPointsSelCosm() const { return GetNPointsSel(AliAlgAux::kCosm); }
  //
  //
  bool IsObligatory(int tp) const { return fObligatory[tp]; }
  bool IsObligatoryColl() const { return IsObligatory(AliAlgAux::kColl); }
  bool IsObligatoryCosm() const { return IsObligatory(AliAlgAux::kCosm); }
  void SetObligatory(int tp, bool v = true);
  void SetObligatoryColl(bool v = true) { SetObligatory(AliAlgAux::kColl, v); }
  void SetObligatoryCosm(bool v = true) { SetObligatory(AliAlgAux::kCosm, v); }
  //
  void AddAutoConstraints() const;
  void ConstrainOrphans(const double* sigma, const char* match = 0);

  virtual void WritePedeInfo(FILE* parOut, const Option_t* opt = "") const;
  virtual void WriteCalibrationResults() const;
  virtual void WriteAlignmentResults() const;
  //
  float* GetParVals() const { return fParVals; }
  double GetParVal(int par) const { return fParVals ? fParVals[par] : 0; }
  double GetParErr(int par) const { return fParErrs ? fParErrs[par] : 0; }
  int GetParLab(int par) const { return fParLabs ? fParLabs[par] : 0; }
  //
  void SetParVals(int npar, double* vl, double* er);
  void SetParVal(int par, double v = 0) { fParVals[par] = v; }
  void SetParErr(int par, double e = 0) { fParErrs[par] = e; }
  //
  int GetFirstParGloID() const { return fFirstParGloID; }
  int GetParGloID(int par) const { return fFirstParGloID + par; }
  void SetFirstParGloID(int id) { fFirstParGloID = id; }
  //
 protected:
  void SortSensors();
  void CalcFree(bool condFree = false);
  //
  // ------- dummies ---------
  AliAlgDet(const AliAlgDet&);
  AliAlgDet& operator=(const AliAlgDet&);
  //
 protected:
  //
  int fNDOFs;       // number of DOFs free
  int fVolIDMin;    // min volID for this detector (for sensors only)
  int fVolIDMax;    // max volID for this detector (for sensors only)
  int fNSensors;    // number of sensors (i.e. volID's)
  int* fSID2VolID;  //[fNSensors] table of conversion from VolID to sid
  int fNProcPoints; // total number of points processed
  //
  // Detector specific calibration degrees of freedom
  int fNCalibDOF;     // number of calibDOFs for detector (preset)
  int fNCalibDOFFree; // number of calibDOFs for detector (preset)
  uint64_t fCalibDOF; // status of calib dof
  int fFirstParGloID; // ID of the 1st parameter in the global results array
  float* fParVals;    //! values of the fitted params
  float* fParErrs;    //! errors of the fitted params
  int* fParLabs;      //! labels for parameters
  //
  // Track selection
  bool fDisabled[AliAlgAux::kNTrackTypes];         // detector disabled/enabled in the track
  bool fObligatory[AliAlgAux::kNTrackTypes];       // detector must be present in the track
  uint64_t fTrackFlagSel[AliAlgAux::kNTrackTypes]; // flag for track selection
  int fNPointsSel[AliAlgAux::kNTrackTypes];        // min number of points to require
  //
  int fUseErrorParam;  // signal that points need to be updated using track info, 0 - no
  double fAddError[2]; // additional error increment for measurement
  TObjArray fSensors;  // all sensors of the detector
  TObjArray fVolumes;  // all volumes of the detector
  //
  // this is transient info
  int fNPoints;           //! number of points from this detector
  int fPoolNPoints;       //! number of points in the pool
  int fPoolFreePointID;   //! id of the last free point in the pool
  TObjArray fPointsPool;  //! pool of aligment points
                          //
  AliAlgSteer* fAlgSteer; // pointer to alignment steering object
  //
  ClassDef(AliAlgDet, 1); // base class for detector global alignment
};

//FIXME(milettri): needs AliESDtrack
////_____________________________________________________
//inline bool AliAlgDet::CheckFlags(const AliESDtrack* trc, int trtype) const
//{
//  // check if flags are ok
//  return (trc->GetStatus() & fTrackFlagSel[trtype]) == fTrackFlagSel[trtype];
//}
} // namespace align
} // namespace o2
#endif
