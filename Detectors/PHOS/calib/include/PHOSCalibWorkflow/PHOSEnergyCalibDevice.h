// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_PHOSENERGY_CALIBDEV_H
#define O2_CALIBRATION_PHOSENERGY_CALIBDEV_H

/// @file   PHOSEnergyCalibDevice.h
/// @brief  Device to collect histos and digits for PHOS energy and time calibration.

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ProcessingContext.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsPHOS/Cluster.h"
#include "DataFormatsPHOS/BadChannelsMap.h"
#include "DataFormatsPHOS/CalibParams.h"
#include "PHOSCalibWorkflow/PHOSEnergyCalibrator.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSEnergyCalibDevice : public o2::framework::Task
{
 public:
  explicit PHOSEnergyCalibDevice(bool useCCDB, std::string path, std::string digitspath) : mUseCCDB(useCCDB), mCCDBPath(path), mdigitsfilename(digitspath) {}

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 protected:
 private:
  static constexpr short kMaxCluInEvent = 64; /// maximal number of clusters per event to separate digits from them (6 bits in digit map)
  bool mUseCCDB = false;
  std::string mCCDBPath{"http://ccdb-test.cern.ch:8080"}; ///< CCDB server path
  std::string mdigitsfilename = "./CalibDigits.root";
  long mRunStartTime = 0; /// start time of the run (sec)
  float mPtMin = 1.5;     /// minimal energy to fill inv. mass histo
  float mEminHGTime = 1.5;
  float mEminLGTime = 5.;
  std::unique_ptr<PHOSEnergyCalibrator> mCalibrator; /// Agregator of calibration TimeFrameSlots
  std::unique_ptr<BadChannelsMap> mBadMap;           /// Latest bad channels map
  std::unique_ptr<CalibParams> mCalibParams;         /// Latest bad channels map
  ClassDefNV(PHOSEnergyCalibDevice, 1);
};

o2::framework::DataProcessorSpec getPHOSEnergyCalibDeviceSpec(bool useCCDB, std::string path, std::string digitspath);
} // namespace phos
} // namespace o2

#endif
