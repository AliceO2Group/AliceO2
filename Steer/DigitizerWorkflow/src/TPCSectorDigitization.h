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

#ifndef STEER_DIGITIZERWORKFLOW_TPCSECTORDIGITIZATION_H_
#define STEER_DIGITIZERWORKFLOW_TPCSECTORDIGITIZATION_H_

#include <functional>
#include <vector>
#include <RtypesCore.h>
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsTPC/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TPCSimulation/CommonMode.h"
#include "TPCCalibration/VDriftHelper.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/InputSpec.h"

class TChain;

namespace o2
{
namespace steer
{
class DigitizationContext;
}
namespace framework
{
class InitContext;
class ProcessingContext;
} // namespace framework
namespace tpc
{
class Digitizer;

/// Output buffers of one flush of a sector
struct SectorFlushBuffers {
  std::vector<Digit> digits;
  dataformats::MCTruthContainer<MCCompLabel> labels;
  std::vector<CommonMode> commonMode;
};

/// Digitize all collisions of one sector of a timeframe, calling onFlush after each flush
/// \return the digit groups, one per collision part in triggered mode, one for the timeframe in continuous mode
std::vector<dataformats::RangeReference<int, int>> digitizeSector(Digitizer& digitizer, const steer::DigitizationContext& context,
                                                                  const std::vector<TChain*>& simChains, int sector, ULong64_t timeframe,
                                                                  SectorFlushBuffers& buffers, const std::function<void()>& onFlush);

/// CCDB conditions and calibration files of the TPC digitizer devices
class TPCDigitizerConditions
{
 public:
  /// options common to the TPC digitizer devices
  static std::vector<framework::ConfigParamSpec> options();
  /// CCDB inputs common to the TPC digitizer devices
  static void addInputs(std::vector<framework::InputSpec>& inputs, int distortionType);

  void init(framework::InitContext& ic, int distortionType);
  /// pass a CCDB object to the digitizer; true if it was a TPC condition
  bool finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj, Digitizer& digitizer);
  /// fetch the conditions of a timeframe and prepare the digitizer, which then holds the distortions
  /// \param setVDrift called with the drift velocity and time offset when they changed
  void update(framework::ProcessingContext& pc, Digitizer& digitizer, const std::function<void(float, float)>& setVDrift);

 private:
  VDriftHelper mVDriftHelper{};
  int mDistortionType = 0;
  bool mUseCalibrationsFromCCDB = false;
  bool mRecalcDistortions = false;
  float mMeanLumiDistortions = -1;
  float mMeanLumiDistortionsDerivative = -1;
};

} // namespace tpc
} // namespace o2

#endif
