// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_PHOSHGLGRATIO_CALIBRATOR_H
#define O2_CALIBRATION_PHOSHGLGRATIO_CALIBRATOR_H

/// @file   HGLGRatioCalibSpec.h
/// @brief  Device to calculate PHOS HG/LG ratio

#include "Framework/Task.h"
// #include "Framework/ConfigParamRegistry.h"
// #include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "PHOSReconstruction/CaloRawFitter.h"
#include "PHOSBase/Mapping.h"
#include "PHOSCalib/CalibParams.h"
#include "TH2.h"

using namespace o2::framework;

namespace o2
{
namespace phos
{

class PHOSHGLGRatioCalibDevice : public o2::framework::Task
{

  union PairAmp {
    uint32_t mDataWord;
    struct {
      uint32_t mHGAmp : 16; ///< Bits  0 - 15: HG amplitude in channel
      uint32_t mLGAmp : 16; ///< Bits 16 - 25: LG amplitude in channel
    };
  };

 public:
  explicit PHOSHGLGRatioCalibDevice(bool useCCDB, bool forceUpdate, std::string path) : mUseCCDB(useCCDB), mForceUpdate(forceUpdate), mPath(path) {}
  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 protected:
  void sendOutput(DataAllocator& output);
  void fillRatios();
  void calculateRatios();
  void checkRatios();

  // void evaluateMeans

 private:
  bool mUseCCDB = false;
  bool mForceUpdate = false;                                  /// Update CCDB even if difference to current is large
  bool mUpdateCCDB = true;                                    /// set is close to current and can update it
  std::string mPath{"./"};                                    /// path and name of file with collected histograms
  std::unique_ptr<CalibParams> mCalibParams;                  /// Final calibration object
  short mMinLG = 20;                                          /// minimal LG ampl used in ratio
  short minimalStatistics = 100;                              /// minimal statistics per channel
  std::map<short, PairAmp> mMapPairs;                         /// HG/LG pair
  std::unique_ptr<Mapping> mMapping;                          /// Mapping
  std::unique_ptr<CaloRawFitter> mRawFitter;                  /// Sample fitting class
  std::unique_ptr<CalibParams> mCalibObject;                  /// Final calibration object
  std::unique_ptr<TH2F> mhRatio;                              /// Histogram with ratios
  std::array<float, o2::phos::Mapping::NCHANNELS> mRatioDiff; /// Ratio variation wrt previous map
};

DataProcessorSpec getHGLGRatioCalibSpec(bool useCCDB, bool forceUpdate, std::string path);

} // namespace phos
} // namespace o2

#endif
