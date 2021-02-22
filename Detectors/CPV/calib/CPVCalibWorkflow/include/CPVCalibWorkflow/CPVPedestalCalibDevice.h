// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_CPVPEDESTALS_CALIBRATOR_H
#define O2_CALIBRATION_CPVPEDESTALS_CALIBRATOR_H

/// @file   PedestalCalibSpec.h
/// @brief  Device to calculate CPV pedestals

#include "Framework/Task.h"
// #include "Framework/ConfigParamRegistry.h"
// #include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "CPVCalib/Pedestals.h"
#include "CPVBase/Geometry.h"
#include "TH2.h"

using namespace o2::framework;

namespace o2
{
namespace cpv
{

class CPVPedestalCalibDevice : public o2::framework::Task
{

 public:
  explicit CPVPedestalCalibDevice(bool useCCDB, bool forceUpdate, std::string path) : mUseCCDB(useCCDB), mForceUpdate(forceUpdate), mPath(path) {}
  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 protected:
  void sendOutput(DataAllocator& output);

  void calculatePedestals();
  void checkPedestals();

 private:
  bool mUseCCDB = false;
  bool mForceUpdate = false;                                 /// Update CCDB even if difference to current is large
  bool mUpdateCCDB = true;                                   /// set is close to current and can update it
  std::string mPath{"./"};                                   ///< path and name of file with collected histograms
  std::unique_ptr<Pedestals> mPedestals;                     /// Final calibration object
  std::unique_ptr<TH2F> mMean;                               /// Mean values in High Gain channels
  std::array<short, o2::cpv::Geometry::kNCHANNELS> mPedDiff; /// Pedestal variation wrt previous map
  ClassDefNV(CPVPedestalCalibDevice, 1);
};

o2::framework::DataProcessorSpec getPedestalCalibSpec(bool useCCDB, bool forceUpdate, std::string path);

} // namespace cpv
} // namespace o2

#endif
