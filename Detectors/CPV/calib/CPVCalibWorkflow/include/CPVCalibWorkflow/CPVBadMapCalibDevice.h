// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_CPVBADMAP_CALIBRATOR_H
#define O2_CALIBRATION_CPVBADMAP_CALIBRATOR_H

/// @file   BadMapCalibSpec.h
/// @brief  Device to calculate CPV bad map

#include "Framework/Task.h"
// #include "Framework/ConfigParamRegistry.h"
// #include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "CPVCalib/BadChannelMap.h"
#include "CPVBase/Geometry.h"
#include "TH2.h"

using namespace o2::framework;

namespace o2
{
namespace cpv
{

class CPVBadMapCalibDevice : public o2::framework::Task
{

 public:
  explicit CPVBadMapCalibDevice(bool useCCDB, bool forceUpdate, std::string path, short m) : mUseCCDB(useCCDB), mForceUpdate(forceUpdate), mPath(path), mMethod(m) {}
  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 protected:
  void sendOutput(DataAllocator& output);

  bool differFromCurrent();

 private:
  bool mUseCCDB = false;     /// Use CCDB for comparison and update
  bool mForceUpdate = false; /// Update CCDB even if difference to current is large
  bool mUpdateCCDB = true;   /// set is close to current and can update it
  short mMethod = 0;
  std::string mPath{"./"};                                  ///< path and name of file with collected histograms
  std::unique_ptr<BadChannelMap> mBadMap;                   /// Final calibration object
  std::array<char, o2::cpv::Geometry::kNCHANNELS> mMapDiff; /// difference between new and old map
};

o2::framework::DataProcessorSpec getBadMapCalibSpec(bool useCCDB, bool forceUpdate, std::string path, short method);

} // namespace cpv
} // namespace o2

#endif
