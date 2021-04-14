// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AlignableDetector.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Base class for detector: wrapper for set of volumes

#ifndef ALIGNABLEDETECTOR_H
#define ALIGNABLEDETECTOR_H

#include <TNamed.h>
#include <TObjArray.h>
#include <stdio.h>
#include "Align/utils.h"
#include "Align/AlignmentTrack.h"
#include "Align/DOFStatistics.h"
#include "Align/AlignmentPoint.h"
#include "Align/AlignableSensor.h"
#include "Align/AlignableVolume.h"
#include "Align/Controller.h"
//#include "AliESDtrack.h"
//class AliTrackPointArray;

class TH1;

namespace o2
{
namespace align
{

//TODO(milettri) : fix possibly incompatible Detector IDs of O2 and AliROOT
class AlignableDetector : public TNamed
{
 public:
  enum { kInitGeomDone = BIT(14),
         kInitDOFsDone = BIT(15) };
  enum { kNMaxKalibDOF = 64 };
  //
  AlignableDetector();
  AlignableDetector(const char* name, const char* title = "", int id = -1) : TNamed(name, title) { SetUniqueID(id); };
  virtual ~AlignableDetector();
  int getDetID() const { return GetUniqueID(); }
  detectors::DetID getO2DetID() const { return detectors::DetID(GetUniqueID()); };
  void setDetID(uint32_t tp);
  void SetDetID(detectors::DetID id) { SetUniqueID(id); }
  //
  virtual void cacheReferenceOCDB();
  virtual void acknowledgeNewRun(int run);
  virtual void updateL2GRecoMatrices();
  virtual void applyAlignmentFromMPSol();
  //
  int volID2SID(int vid) const;
  int sID2VolID(int sid) const { return sid < getNSensors() ? mSID2VolID[sid] : -1; } //todo
  int getNSensors() const { return mSensors.GetEntriesFast(); }
  int getNVolumes() const { return mVolumes.GetEntriesFast(); }
  int getVolIDMin() const { return mVolIDMin; }
  int getVolIDMax() const { return mVolIDMax; }
  bool sensorOfDetector(int vid) const { return vid >= mVolIDMin && vid <= mVolIDMax; }
  void setAddError(double y, double z);
  const double* getAddError() const { return mAddError; }
  //
  int getNPoints() const { return mNPoints; }
  //
  void setAlgSteer(Controller* s) { mAlgSteer = s; }
  AlignableSensor* getSensor(int id) const { return (AlignableSensor*)mSensors.UncheckedAt(id); }
  AlignableSensor* getSensorByVolId(int vid) const
  {
    int sid = volID2SID(vid);
    return sid < 0 ? 0 : getSensor(sid);
  }
  AlignableSensor* getSensor(const char* symname) const { return (AlignableSensor*)mSensors.FindObject(symname); }
  AlignableVolume* getVolume(int id) const { return (AlignableVolume*)mVolumes.UncheckedAt(id); }
  AlignableVolume* getVolume(const char* symname) const { return (AlignableVolume*)mVolumes.FindObject(symname); }
  //
  bool ownsDOFID(int id) const;
  AlignableVolume* getVolOfDOFID(int id) const;
  //
  int getDetLabel() const { return (getDetID() + 1) * 1000000; }
  void setFreeDOF(int dof);
  void fixDOF(int dof);
  void setFreeDOFPattern(uint64_t pat)
  {
    mCalibDOF = pat;
    calcFree();
  }
  bool isFreeDOF(int dof) const { return (mCalibDOF & (0x1 << dof)) != 0; }
  bool isCondDOF(int dof) const;
  uint64_t getFreeDOFPattern() const { return mCalibDOF; }
  int getNProcessedPoints() const { return mNProcPoints; }
  virtual const char* getCalibDOFName(int) const { return 0; }
  virtual double getCalibDOFVal(int) const { return 0; }
  virtual double getCalibDOFValWithCal(int) const { return 0; }
  //
  virtual int initGeom();
  virtual int assignDOFs();
  virtual void initDOFs();
  virtual void terminate();
  void fillDOFStat(DOFStatistics* dofst = 0) const;
  virtual void addVolume(AlignableVolume* vol);
  virtual void defineVolumes();
  virtual void defineMatrices();
  virtual void Print(const Option_t* opt = "") const;
  //  virtual int ProcessPoints(const AliESDtrack* esdTr, AlignmentTrack* algTrack, bool inv = false); FIXME(milettri): needs AliESDtrack
  virtual void updatePointByTrackInfo(AlignmentPoint* pnt, const trackParam_t* t) const;
  virtual void setUseErrorParam(int v = 0);
  int getUseErrorParam() const { return mUseErrorParam; }
  //
  //  virtual bool AcceptTrack(const AliESDtrack* trc, int trtype) const = 0; FIXME(milettri): needs AliESDtrack
  //  bool CheckFlags(const AliESDtrack* trc, int trtype) const; FIXME(milettri): needs AliESDtrack
  //
  virtual AlignmentPoint* getPointFromPool();
  virtual void resetPool();
  virtual void writeSensorPositions(const char* outFName);
  //
  void setInitGeomDone() { SetBit(kInitGeomDone); }
  bool getInitGeomDone() const { return TestBit(kInitGeomDone); }
  //
  void setInitDOFsDone() { SetBit(kInitDOFsDone); }
  bool getInitDOFsDone() const { return TestBit(kInitDOFsDone); }
  void fixNonSensors();
  void setFreeDOFPattern(uint32_t pat = 0xffffffff, int lev = -1, const char* match = 0);
  void setDOFCondition(int dof, float condErr, int lev = -1, const char* match = 0);
  int selectVolumes(TObjArray* arr, int lev = -1, const char* match = 0);
  //
  int getNDOFs() const { return mNDOFs; }
  int getNCalibDOFs() const { return mNCalibDOF; }
  int getNCalibDOFsFree() const { return mNCalibDOFFree; }
  //
  void setDisabled(int tp, bool v)
  {
    mDisabled[tp] = v;
    setObligatory(tp, !v);
  }
  void setDisabled()
  {
    setDisabledColl();
    setDisabledCosm();
  }
  void setDisabledColl(bool v = true) { setDisabled(utils::Coll, v); }
  void setDisabledCosm(bool v = true) { setDisabled(utils::Cosm, v); }
  bool isDisabled(int tp) const { return mDisabled[tp]; }
  bool isDisabled() const { return IsDisabledColl() && IsDisabledCosm(); }
  bool IsDisabledColl() const { return isDisabled(utils::Coll); }
  bool IsDisabledCosm() const { return isDisabled(utils::Cosm); }
  //
  void setTrackFlagSel(int tp, uint64_t f) { mTrackFlagSel[tp] = f; }
  void setTrackFlagSelColl(uint64_t f) { setTrackFlagSel(utils::Coll, f); }
  void setTrackFlagSelCosm(uint64_t f) { setTrackFlagSel(utils::Cosm, f); }
  uint64_t getTrackFlagSel(int tp) const { return mTrackFlagSel[tp]; }
  uint64_t getTrackFlagSelColl() const { return getTrackFlagSel(utils::Coll); }
  uint64_t getTrackFlagSelCosm() const { return getTrackFlagSel(utils::Cosm); }
  //
  void setNPointsSel(int tp, int n) { mNPointsSel[tp] = n; }
  void setNPointsSelColl(int n) { setNPointsSel(utils::Coll, n); }
  void setNPointsSelCosm(int n) { setNPointsSel(utils::Cosm, n); }
  int getNPointsSel(int tp) const { return mNPointsSel[tp]; }
  int getNPointsSelColl() const { return getNPointsSel(utils::Coll); }
  int getNPointsSelCosm() const { return getNPointsSel(utils::Cosm); }
  //
  //
  bool isObligatory(int tp) const { return mObligatory[tp]; }
  bool isObligatoryColl() const { return isObligatory(utils::Coll); }
  bool isObligatoryCosm() const { return isObligatory(utils::Cosm); }
  void setObligatory(int tp, bool v = true);
  void setObligatoryColl(bool v = true) { setObligatory(utils::Coll, v); }
  void setObligatoryCosm(bool v = true) { setObligatory(utils::Cosm, v); }
  //
  void addAutoConstraints() const;
  void constrainOrphans(const double* sigma, const char* match = 0);

  virtual void writePedeInfo(FILE* parOut, const Option_t* opt = "") const;
  virtual void writeCalibrationResults() const;
  virtual void writeAlignmentResults() const;
  //
  float* getParVals() const { return mParVals; }
  double getParVal(int par) const { return mParVals ? mParVals[par] : 0; }
  double getParErr(int par) const { return mParErrs ? mParErrs[par] : 0; }
  int getParLab(int par) const { return mParLabs ? mParLabs[par] : 0; }
  //
  void setParVals(int npar, double* vl, double* er);
  void setParVal(int par, double v = 0) { mParVals[par] = v; }
  void setParErr(int par, double e = 0) { mParErrs[par] = e; }
  //
  int getFirstParGloID() const { return mFirstParGloID; }
  int getParGloID(int par) const { return mFirstParGloID + par; }
  void setFirstParGloID(int id) { mFirstParGloID = id; }
  //
 protected:
  void sortSensors();
  void calcFree(bool condFree = false);
  //
  // ------- dummies ---------
  AlignableDetector(const AlignableDetector&);
  AlignableDetector& operator=(const AlignableDetector&);
  //
 protected:
  //
  int mNDOFs;       // number of DOFs free
  int mVolIDMin;    // min volID for this detector (for sensors only)
  int mVolIDMax;    // max volID for this detector (for sensors only)
  int mNSensors;    // number of sensors (i.e. volID's)
  int* mSID2VolID;  //[mNSensors] table of conversion from VolID to sid
  int mNProcPoints; // total number of points processed
  //
  // Detector specific calibration degrees of freedom
  int mNCalibDOF;     // number of calibDOFs for detector (preset)
  int mNCalibDOFFree; // number of calibDOFs for detector (preset)
  uint64_t mCalibDOF; // status of calib dof
  int mFirstParGloID; // ID of the 1st parameter in the global results array
  float* mParVals;    //! values of the fitted params
  float* mParErrs;    //! errors of the fitted params
  int* mParLabs;      //! labels for parameters
  //
  // Track selection
  bool mDisabled[utils::NTrackTypes];         // detector disabled/enabled in the track
  bool mObligatory[utils::NTrackTypes];       // detector must be present in the track
  uint64_t mTrackFlagSel[utils::NTrackTypes]; // flag for track selection
  int mNPointsSel[utils::NTrackTypes];        // min number of points to require
  //
  int mUseErrorParam;  // signal that points need to be updated using track info, 0 - no
  double mAddError[2]; // additional error increment for measurement
  TObjArray mSensors;  // all sensors of the detector
  TObjArray mVolumes;  // all volumes of the detector
  //
  // this is transient info
  int mNPoints;          //! number of points from this detector
  int mPoolNPoints;      //! number of points in the pool
  int mPoolFreePointID;  //! id of the last free point in the pool
  TObjArray mPointsPool; //! pool of aligment points
                         //
  Controller* mAlgSteer; // pointer to alignment steering object
  //
  ClassDef(AlignableDetector, 1); // base class for detector global alignment
};

//FIXME(milettri): needs AliESDtrack
////_____________________________________________________
//inline bool AlignableDetector::CheckFlags(const AliESDtrack* trc, int trtype) const
//{
//  // check if flags are ok
//  return (trc->GetStatus() & mTrackFlagSel[trtype]) == mTrackFlagSel[trtype];
//}
} // namespace align
} // namespace o2
#endif
