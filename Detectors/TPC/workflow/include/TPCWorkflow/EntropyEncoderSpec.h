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

#ifndef O2_TPC_ENTROPYENCODERSPEC_H
#define O2_TPC_ENTROPYENCODERSPEC_H
/// @file   EntropyEncoderSpec.h
/// @author Michael Lettrich, Matthias Richter
/// @since  2020-01-16
/// @brief  ProcessorSpec for the TPC cluster entropy encoding

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TPCReconstruction/CTFCoder.h"
#include <TStopwatch.h>

namespace o2
{
namespace tpc
{

class EntropyEncoderSpec : public o2::framework::Task
{
 public:
  EntropyEncoderSpec(bool fromFile) : mCTFCoder(o2::ctf::CTFCoderBase::OpType::Encoder), mFromFile(fromFile)
  {
    mTimer.Stop();
    mTimer.Reset();
  }
  ~EntropyEncoderSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  o2::tpc::CTFCoder mCTFCoder;
  bool mFromFile = false;
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getEntropyEncoderSpec(bool inputFromFile);

} // end namespace tpc
} // end namespace o2

#endif // O2_TPC_ENTROPYENCODERSPEC_H
