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

/// @file   EntropyDecoderSpec.h
/// @brief  Convert CTF (EncodedBlocks) to FT0 digit/channels strean

#ifndef O2_TOF_ENTROPYDECODER_SPEC
#define O2_TOF_ENTROPYDECODER_SPEC

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TOFReconstruction/CTFCoder.h"
#include <TStopwatch.h>

namespace o2
{
namespace tof
{

class EntropyDecoderSpec : public o2::framework::Task
{
 public:
  EntropyDecoderSpec();
  ~EntropyDecoderSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  o2::tof::CTFCoder mCTFCoder;
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getEntropyDecoderSpec();

} // namespace tof
} // namespace o2

#endif
