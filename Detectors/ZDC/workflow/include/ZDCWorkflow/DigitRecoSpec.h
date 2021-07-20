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
/// @brief  Convert ZDC data to CTF (EncodedBlocks)
/// @author pietro.cortese@cern.ch

#ifndef O2_ZDC_DIGITRECO_SPEC
#define O2_ZDC_DIGITRECO_SPEC

#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ZDCReconstruction/DigiReco.h"
#include <TStopwatch.h>

namespace o2
{
namespace zdc
{

class DigitRecoSpec : public o2::framework::Task
{
 public:
  DigitRecoSpec();
  DigitRecoSpec(const int verbosity, const bool debugOut);
  ~DigitRecoSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  DigiReco mDR;                                           // Reconstruction object
  std::string mccdbHost{"http://ccdb-test.cern.ch:8080"}; // Alternative ccdb server
  int mVerbosity = 0;                                     // Verbosity level during recostruction
  bool mDebugOut = false;                                 // Save temporary reconstruction structures on root file
  bool mInitialized = false;                              // Connect once to CCDB during initialization
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getDigitRecoSpec(const int verbosity, const bool enableDebugOut);

} // namespace zdc
} // namespace o2

#endif
