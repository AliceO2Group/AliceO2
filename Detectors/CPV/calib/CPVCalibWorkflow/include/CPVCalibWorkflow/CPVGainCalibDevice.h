// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_CPVGAINS_CALIBRATOR_H
#define O2_CALIBRATION_CPVGAINS_CALIBRATOR_H

/// @file   CPVGainCalibSpec.h
/// @brief  Device to calculate CPV gains

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "CPVCalib/CalibParams.h"
#include "CPVBase/Geometry.h"
#include "TH2.h"
#include <string>

using namespace o2::framework;

namespace o2
{
namespace cpv
{

class CPVGainCalibDevice : public o2::framework::Task
{

 public:
  explicit CPVGainCalibDevice(bool useCCDB, bool forceUpdate, std::string path) : mUseCCDB(useCCDB), mForceUpdate(forceUpdate), mPath(path) {}
  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 protected:
  void sendOutput(DataAllocator& output);
  void calculateGains();
  void checkGains();

 private:
  static constexpr int kMinimalStatistics = 150; /// Minimal statistics per channel

  bool mUseCCDB = false;     /// Use CCDB for comparison and update
  bool mForceUpdate = false; /// Update CCDB even if difference to current is large
  bool mUpdateCCDB = true;   /// set is close to current and can update it

  std::string mPath{"./"};                                     ///< path and name of file with collected histograms
  std::unique_ptr<CalibParams> mCalibParams;                   /// Final calibration object
  std::unique_ptr<TH2F> mMean;                                 /// Mean values in High Gain channels
  std::array<short, o2::cpv::Geometry::kNCHANNELS> mGainRatio; /// Gain variation wrt previous map
};

o2::framework::DataProcessorSpec getGainCalibSpec(bool useCCDB, bool forceUpdate, std::string path);

} // namespace cpv
} // namespace o2

#endif
