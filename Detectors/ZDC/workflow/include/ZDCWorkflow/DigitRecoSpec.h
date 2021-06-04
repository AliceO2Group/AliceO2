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
  ~DigitRecoSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  DigiReco mDR;
  TStopwatch mTimer;
  bool mInitialized = false;
};

/// create a processor spec
framework::DataProcessorSpec getDigitRecoSpec();

} // namespace zdc
} // namespace o2

#endif
