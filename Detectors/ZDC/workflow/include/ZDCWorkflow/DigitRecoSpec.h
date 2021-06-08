// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitRecoSpec.h
/// @brief  Convert ZDC data to CTF (EncodedBlocks)
/// @author pietro.cortese@cern.ch

#ifndef O2_ZDC_DIGITRECO_SPEC
#define O2_ZDC_DIGITRECO_SPEC

#include "FairLogger.h"
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
  DigitRecoSpec(const int verbosity, const bool debugOut, const std::string& ccdbURL);
  ~DigitRecoSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  DigiReco mDR;              // Reconstruction object
  std::string mccdbHost;     // Alternative ccdb server
  int mVerbosity = 0;        // Verbosity level during recostruction
  bool mDebugOut = false;    // Save temporary reconstruction structures on root file
  bool mInitialized = false; // Connect once to CCDB during initialization
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getDigitRecoSpec(const int verbosity, const bool enableDebugOut, const std::string ccdbURL);

} // namespace zdc
} // namespace o2

#endif
