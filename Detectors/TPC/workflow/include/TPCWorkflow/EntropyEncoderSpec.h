// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  EntropyEncoderSpec(bool fromFile) : mFromFile(fromFile)
  {
    mTimer.Stop();
    mTimer.Reset();
  }
  ~EntropyEncoderSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

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
