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
//class PRFWidth;
//class TrapConfig;
class TRDGeometry; //

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
  void setCCDB(int calibrationobjecttype);
  void setCCDBForSimulation() { setCCDB(kSimulation); };
  void setCCDBForReconstruction() { setCCDB(kReconstruction); };
  void setCCDBForCalibration() { setCCDB(kCalibration); };
  //// cant send auto to setCCDB, as a temporary measure, until I can figure something else out. There are 3 formats of getting CCDB parameters defined by the enum at the top.
  //
  //  float getGainMeanRMS();
  //  float getT0MeanRMS();
  double getVDrift(long timestamp, int roc, int col, int row) const;
  double getT0(long timestamp, int roc, int col, int row) const;
  double getExB(long timestamp, int roc, int col, int row) const;
  double getGainFactor(long timestamp, int roc, int col, int row) const;

 protected:
  int mTimeStamp; //run number of related to the current calibration.
  // here we have pointers to all the incoming calibrations, not all of them will be valid
  // this will be dictated by the DPL and what it provides. (if I understand things correctly)
  // pointers to relevant incoming classes will sit here and thereby provide the correct api
  // abstracting all the tedious details from users. Most importantly we can change things with
  // out them knowing.
  std::shared_ptr<ChamberCalibrations> mChamberCalibrations;
  std::shared_ptr<LocalVDrift> mLocalVDrift;
  std::shared_ptr<LocalT0> mLocalT0;
  std::shared_ptr<LocalGainFactor> mLocalGainFactor;
  std::shared_ptr<PadNoise> mPadNoise;
  std::shared_ptr<ChamberStatus> mChamberStatus;
  std::shared_ptr<PadStatus> mPadStatus;
  std::shared_ptr<ChamberNoise> mChamberNoise;
  //this will probably get extended (26/09/2019), the list of pointers above.
  //std::shared_ptr<TrapConfig> mTrapConfig;
  //std::shared_ptr<PRFWidth> mPRDWidth
  //std::shared_ptr<OnlineGainFactors> mOnlineGainFactors;
  //
};
} // namespace trd
} // namespace o2
#endif
