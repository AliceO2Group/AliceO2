// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_PHOSPEDESTALS_CALIBRATOR_H
#define O2_CALIBRATION_PHOSPEDESTALS_CALIBRATOR_H

/// @file   PedestalCalibSpec.h
/// @brief  Device to calculate PHOS pedestals

#include "Framework/Task.h"
// #include "Framework/ConfigParamRegistry.h"
// #include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "PHOSReconstruction/CaloRawFitter.h"
#include "PHOSBase/Mapping.h"
#include "PHOSCalib/Pedestals.h"
#include "TH2.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSPedestalCalibDevice : public o2::framework::Task
{

 public:
  explicit PHOSPedestalCalibDevice(bool useCCDB, bool forceUpdate, std::string path) : mUseCCDB(useCCDB), mForceUpdate(forceUpdate), mPath(path) {}

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 protected:
  void sendOutput(DataAllocator& output);
  // void evaluateMeans
  void calculatePedestals();
  void checkPedestals();

 private:
  bool mUseCCDB = false;
  bool mForceUpdate = false;                                  /// Update CCDB even if difference to current is large
  bool mUpdateCCDB = true;                                    /// set is close to current and can update it
  std::string mPath{"./"};                                    ///< path and name of file with collected histograms
  std::unique_ptr<Pedestals> mPedestals;                      /// Final calibration object
  std::unique_ptr<Mapping> mMapping;                          /// Mapping
  std::unique_ptr<CaloRawFitter> mRawFitter;                  /// Sample fitting class
  std::unique_ptr<TH2F> mMeanHG;                              /// Mean values in High Gain channels
  std::unique_ptr<TH2F> mMeanLG;                              /// RMS of values in High Gain channels
  std::unique_ptr<TH2F> mRMSHG;                               /// Mean values in Low Gain channels
  std::unique_ptr<TH2F> mRMSLG;                               /// RMS of values in Low Gain channels
  std::array<short, o2::phos::Mapping::NCHANNELS> mPedHGDiff; /// Pedestal variation wrt previous map
  std::array<short, o2::phos::Mapping::NCHANNELS> mPedLGDiff; /// Pedestal variation wrt previous map
};

o2::framework::DataProcessorSpec getPedestalCalibSpec(bool useCCDB, bool forceUpdate, std::string path);
} // namespace phos
} // namespace o2

#endif
