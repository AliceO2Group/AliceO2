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

/// @file   AlignableDetector.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Base class for detector: wrapper for set of volumes

#ifndef ALIGNABLEDETECTOR_H
#define ALIGNABLEDETECTOR_H

#include "DetectorsCommonDataFormats/DetID.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include <TObjArray.h>
#include <cstdio>
#include "Align/DOFSet.h"
#include "Align/utils.h"
#include "Align/AlignmentTrack.h"
#include "Align/AlignmentPoint.h"
#include "Align/AlignableSensor.h"
#include "Align/AlignableVolume.h"

class TH1;

namespace o2
{
namespace align
{
using GIndex = o2::dataformats::VtxTrackIndex;
class Controller;

//TODO(milettri) : fix possibly incompatible Detector IDs of O2 and AliROOT
class AlignableDetector : public DOFSet
{
 public:
  using DetID = o2::detectors::DetID;

  enum { kInitGeomDone = BIT(14),
         kInitDOFsDone = BIT(15) };
  enum { kNMaxKalibDOF = 64 };
  //
  AlignableDetector() = default;
  AlignableDetector(DetID id, Controller* ctr);
  ~AlignableDetector() override;

  auto getDetID() const { return mDetID; }
  auto getName() const { return mDetID.getName(); }
  //
  virtual void cacheReferenceCCDB();
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
  AlignableSensor* getSensor(int id) const { return (AlignableSensor*)mSensors.UncheckedAt(id); }
  AlignableSensor* getSensorByVolId(int vid) const
  {
    int sid = volID2SID(vid);
    return sid < 0 ? nullptr : getSensor(sid);
  }
  AlignableSensor* getSensor(const char* symname) const { return (AlignableSensor*)mSensors.FindObject(symname); }
  AlignableVolume* getVolume(int id) const { return (AlignableVolume*)mVolumes.UncheckedAt(id); }
  AlignableVolume* getVolume(const char* symname) const { return (AlignableVolume*)mVolumes.FindObject(symname); }
  //
  bool ownsDOFID(int id) const;
  AlignableVolume* getVolOfDOFID(int id) const;
  //
  int getDetLabel() const { return (getDetID() + 1) * 100000; }
  int getSensLabel(int i) const { return getDetLabel() + i + 1; }
  int getNonSensLabel(int i) const { return getDetLabel() + i + 50001; }
  int getSensID(int lbl) const { return (lbl % 100000) < 50001 ? (lbl % 100000) - 1 : -1; }
  int getNonSensID(int lbl) const { return (lbl % 100000) < 50001 ? -1 : (lbl % 100000) - 50001; }

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
  virtual const char* getCalibDOFName(int) const { return nullptr; }
  virtual double getCalibDOFVal(int) const { return 0; }
  virtual double getCalibDOFValWithCal(int) const { return 0; }
  //
  virtual int initGeom();
  virtual int assignDOFs();
  virtual void initDOFs();
  virtual void terminate();
  virtual void addVolume(AlignableVolume* vol);
  virtual void defineVolumes();
  virtual void defineMatrices();
  void Print(const Option_t* opt = "") const override;

  virtual void reset();
  virtual int processPoints(GIndex gid, int npntCut = 0, bool inv = false);
  virtual bool prepareDetectorData() { return true; }

  virtual void updatePointByTrackInfo(AlignmentPoint* pnt, const trackParam_t* t) const;
  virtual void setUseErrorParam(int v = 0);
  int getUseErrorParam() const { return mUseErrorParam; }
  //
  //  virtual bool AcceptTrack(const AliESDtrack* trc, int trtype) const = 0; FIXME(milettri): needs AliESDtrack
  //  bool CheckFlags(const AliESDtrack* trc, int trtype) const; FIXME(milettri): needs AliESDtrack
  //
  virtual void writeSensorPositions(const char* outFName);
  //
  void setInitGeomDone() { SetBit(kInitGeomDone); }
  bool getInitGeomDone() const { return TestBit(kInitGeomDone); }
  //
  void setInitDOFsDone() { SetBit(kInitDOFsDone); }
  bool getInitDOFsDone() const { return TestBit(kInitDOFsDone); }
  void fixNonSensors();
  void setFreeDOFPattern(uint32_t pat = 0xffffffff, int lev = -1, const std::string& regexStr = "");
  void setDOFCondition(int dof, float condErr, int lev = -1, const std::string& regexStr = "");
  int selectVolumes(std::vector<AlignableVolume*> cont, int lev = -1, const std::string& regexStr = "");
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
  void constrainOrphans(const double* sigma, const char* match = nullptr);

  virtual void writePedeInfo(FILE* parOut, const Option_t* opt = "") const;
  virtual void writeLabeledPedeResults(FILE* parOut) const;
  virtual void writeCalibrationResults() const;
  virtual void writeAlignmentResults() const;
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
  DetID mDetID{}; // detector ID
  bool mInitDone = false;
  int mVolIDMin = -1;        // min volID for this detector (for sensors only)
  int mVolIDMax = -1;        // max volID for this detector (for sensors only)
  int mNSensors = 0;         // number of sensors (i.e. volID's)
  int* mSID2VolID = nullptr; //[mNSensors] table of conversion from VolID to sid
  int mNProcPoints = 0;      // total number of points processed
  //
  // Detector specific calibration degrees of freedom
  uint64_t mCalibDOF = 0; // status of calib dof
  //
  // Track selection
  bool mDisabled[utils::NTrackTypes] = {};         // detector disabled/enabled in the track
  bool mObligatory[utils::NTrackTypes] = {};       // detector must be present in the track
  int mNPointsSel[utils::NTrackTypes] = {};        // min number of points to require
  //
  int mUseErrorParam = 0;   // signal that points need to be updated using track info, 0 - no
  double mAddError[2] = {}; // additional error increment for measurement
  TObjArray mSensors;  // all sensors of the detector
  TObjArray mVolumes;  // all volumes of the detector
  //
  // this is transient info
  int mNPoints = 0; //! number of points from this detector
  //
  ClassDefOverride(AlignableDetector, 1); // base class for detector global alignment
};

} // namespace align
} // namespace o2
#endif
