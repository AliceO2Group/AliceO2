// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_CALIBRATIONS_H
#define O2_TRD_CALIBRATIONS_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD calibration class for parameters which are saved frequently(/run)    //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  A wrapper api class to provide a user intuitive interface to the
//  underlying calibrations classes and ccdb and make life easy for all
//  bold statement, lets see if we get there.
//
//  Questions :
//  1. Its not clear to me yet what happens if I query an object that
//      is not defined as input in the dpl for the device we are using.
//
//  This pulls in functions from various subordinate classes like CalPad CalROC
//  CalDet etc. and renames them to clearly reflect their applicability,
//
//  This class does *not* store pointers to any subordinate objects as that
//  would imply caching. At each request the underlying object stored in
//  CCDB is sought, either directly or for that which is provided by DPL. The
//  latter still being unclear (17/09/2019)
//  BE AWARE THIS IS LIKELY TO CHANGE BEFORE START OF RUN 3 (25/09/2019)
///////////////////////////////////////////////////////////////////////////////

#include <memory>
#include "TRDBase/TRDGeometry.h"
#include "TRDBase/ChamberCalibrations.h"
#include "TRDBase/LocalVDrift.h"
#include "TRDBase/LocalT0.h"
#include "TRDBase/LocalGainFactor.h"
#include "TRDBase/ChamberStatus.h"
#include "TRDBase/ChamberNoise.h"
#include "TRDBase/PadNoise.h"
#include "TRDBase/PadStatus.h"

class TRDGeometry;

namespace o2
{
namespace trd
{

class Calibrations
{
  enum {
    kSimulation = 1,
    kReconstruction = 2,
    kCalibration = 3
  };

 public:
  Calibrations() = default;
  ~Calibrations() = default;
  //
  int const getTimeStamp() { return mTimeStamp; }
  void setTimeStamp(long timestamp) { mTimeStamp = timestamp; }
  void setCCDB(int calibrationobjecttype, long timestamp);
  void setCCDBForSimulation(long timestamp) { setCCDB(kSimulation, timestamp); };
  void setCCDBForReconstruction(long timestamp) { setCCDB(kReconstruction, timestamp); };
  void setCCDBForCalibration(long timestamp) { setCCDB(kCalibration, timestamp); };
  //
  double getVDrift(int roc, int col, int row) const;
  double getT0(int roc, int col, int row) const;
  double getExB(int roc) const;
  double getGainFactor(int roc, int col, int row) const;
  double getPadGainFactor(int roc, int col, int row) const;

  //methods extracted from PadStatus
  bool isPadMasked(int det, int col, int row) const { return mPadStatus->isMasked(det, col, row); }
  bool isPadBridgedLeft(int det, int col, int row) const { return mPadStatus->isBridgedLeft(det, col, row); }
  bool isPadBridgedRight(int det, int col, int row) const { return mPadStatus->isBridgedRight(det, col, row); }
  bool isPadNotConnected(int det, int col, int row) const { return mPadStatus->isNotConnected(det, col, row); };

  //methods extracted from ChamberStatus
  bool isChamberGood(int det) const { return mChamberStatus->isGood(det); }
  bool isChamberNoData(int det) const { return mChamberStatus->isNoData(det); };
  bool isHalfChamberNoData(int det, int side) const { return side > 0 ? isNoDataSideA(det) : isNoDataSideB(det); };
  bool isNoDataSideA(int det) const { return mChamberStatus->isNoDataSideA(det); }
  bool isNoDataSideB(int det) const { return mChamberStatus->isNoDataSideB(det); }
  bool isChamberBadlyCalibrated(int det) const { return mChamberStatus->isBadCalibrated(det); }
  bool isChamberNotCalibrated(int det) const { return mChamberStatus->isNotCalibrated(det); }
  char getChamberStatusRaw(int det) const { return mChamberStatus->getStatus(det); }

 protected:
  long mTimeStamp; //run number of related to the current calibration.
  // here we have pointers to all the incoming calibrations, not all of them will be valid
  // this will be dictated by the DPL and what it provides. (if I understand things correctly)
  // pointers to relevant incoming classes will sit here and thereby provide the correct api
  // abstracting all the tedious details from users. Most importantly we can change things with
  // out the users knowing.
  // I assume at some point the ccdb will provide shared pointers, but for now its raw pointers.
  /* std::shared_ptr<ChamberCalibrations> mChamberCalibrations;
  std::shared_ptr<LocalVDrift> mLocalVDrift;
  std::shared_ptr<LocalT0> mLocalT0;
  std::shared_ptr<LocalGainFactor> mLocalGainFactor;
  std::shared_ptr<PadNoise> mPadNoise;
  std::shared_ptr<ChamberStatus> mChamberStatus;
  std::shared_ptr<PadStatus> mPadStatus;
  std::shared_ptr<ChamberNoise> mChamberNoise; */
  ChamberCalibrations* mChamberCalibrations;
  LocalVDrift* mLocalVDrift;
  LocalT0* mLocalT0;
  LocalGainFactor* mLocalGainFactor;
  PadNoise* mPadNoise;
  ChamberStatus* mChamberStatus;
  PadStatus* mPadStatus;
  ChamberNoise* mChamberNoise;
  //std::shared_ptr<OnlineGainFactors> mOnlineGainFactors;
  //
};
} // namespace trd
} // namespace o2
#endif
