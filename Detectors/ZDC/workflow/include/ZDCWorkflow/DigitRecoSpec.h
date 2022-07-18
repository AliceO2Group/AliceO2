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

/// @file   DigitRecoSpec.h
/// @brief  Run ZDC digits reconstruction
/// @author pietro.cortese@cern.ch

#ifndef O2_ZDC_DIGITRECO_SPEC
#define O2_ZDC_DIGITRECO_SPEC

#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataAllocator.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Task.h"
#include "ZDCReconstruction/DigiReco.h"
#include <TStopwatch.h>
#include "CommonUtils/NameConf.h"

namespace o2
{
namespace zdc
{

class DigitRecoSpec : public o2::framework::Task
{
 public:
  DigitRecoSpec();
  DigitRecoSpec(const int verbosity, const bool debugOut,
                const bool enableZDCTDCCorr, const bool enableZDCEnergyParam, const bool enableZDCTowerParam, const bool enableBaselineParam);
  ~DigitRecoSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc);
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  DigiReco mWorker;                  // Reconstruction object
  int mVerbosity = 0;                // Verbosity level during recostruction
  int mMaxWave = 0;                  // Maximum number of waveforms in output
  bool mDebugOut = false;            // Save temporary reconstruction structures on root file
  bool mEnableZDCTDCCorr = true;     // Get ZDCTDCCorr object
  bool mEnableZDCEnergyParam = true; // Get ZDCEnergyParam object
  bool mEnableZDCTowerParam = true;  // Get ZDCTowerParam object
  bool mEnableBaselineParam = true;  // Get BaselineParam object
  bool mInitialized = false;         // Connect once to CCDB during initialization
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getDigitRecoSpec(const int verbosity, const bool enableDebugOut,
                                              const bool enableZDCTDCCorr, const bool enableZDCEnergyParam, const bool enableZDCTowerParam, const bool enableBaselineParam);

} // namespace zdc
} // namespace o2

#endif
