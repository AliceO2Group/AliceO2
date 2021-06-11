// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_PHOSTURNON_CALIBDEV_H
#define O2_CALIBRATION_PHOSTURNON_CALIBDEV_H

/// @file   PHOSTurnonCalibDevice.h
/// @brief  Device to calculate PHOS turn-on curves and trigger map

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DataFormatsPHOS/Cell.h"
#include "DataFormatsPHOS/Cluster.h"
#include "DataFormatsPHOS/TriggerMap.h"
#include "PHOSCalibWorkflow/PHOSTurnonCalibrator.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSTurnonCalibDevice : public o2::framework::Task
{
 public:
  explicit PHOSTurnonCalibDevice(bool useCCDB, std::string path) : mUseCCDB(useCCDB), mCCDBPath(path) {}

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 protected:
  bool checkFitResult() { return true; } //TODO!! implement true check

 private:
  bool mUseCCDB = false;
  std::string mCCDBPath{"http://ccdb-test.cern.ch:8080"}; ///< CCDB server path
  long mRunStartTime = 0;                                 /// start time of the run (sec)
  std::unique_ptr<TriggerMap> mTriggerMap;                /// Final calibration object
  std::unique_ptr<PHOSTurnonCalibrator> mCalibrator;      /// Agregator of calibration TimeFrameSlots
};

o2::framework::DataProcessorSpec getPHOSTurnonCalibDeviceSpec(bool useCCDB, std::string path);
} // namespace phos
} // namespace o2

#endif
