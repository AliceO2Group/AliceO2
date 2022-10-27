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

/// @file   EntropyEncoderSpec.h
/// @brief  Convert ZDC data to CTF (EncodedBlocks)
/// @author ruben.shahoyan@cern.ch

#ifndef O2_ZDC_ENTROPYENCODER_SPEC
#define O2_ZDC_ENTROPYENCODER_SPEC

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <TStopwatch.h>
#include "ZDCReconstruction/CTFCoder.h"

namespace o2
{
namespace zdc
{

class EntropyEncoderSpec : public o2::framework::Task
{
 public:
  EntropyEncoderSpec(bool selIR);
  ~EntropyEncoderSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  o2::zdc::CTFCoder mCTFCoder;
  bool mSelIR = false;
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getEntropyEncoderSpec(bool selIR = false);

} // namespace zdc
} // namespace o2

#endif
