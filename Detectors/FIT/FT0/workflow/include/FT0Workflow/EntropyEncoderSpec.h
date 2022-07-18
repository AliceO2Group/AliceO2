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
/// @brief  Convert clusters streams to CTF (EncodedBlocks)

#ifndef O2_FT0_ENTROPYENCODER_SPEC
#define O2_FT0_ENTROPYENCODER_SPEC

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <TStopwatch.h>
#include "FT0Reconstruction/CTFCoder.h"

namespace o2
{
namespace ft0
{

class EntropyEncoderSpec : public o2::framework::Task
{
 public:
  EntropyEncoderSpec();
  ~EntropyEncoderSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  o2::ft0::CTFCoder mCTFCoder;
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getEntropyEncoderSpec();

} // namespace ft0
} // namespace o2

#endif
