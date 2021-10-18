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

/// @file GRPLHCIFfileSpec.h

#ifndef O2_GRP_LHC_IF_FILE_SPEC
#define O2_GRP_LHC_IF_FILE_SPEC

#include "GRPCalibration/LHCIFfileReader.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPLHCIFData.h"

using namespace o2::framework;

namespace o2
{
namespace grp
{
class GRPLHCIFfileProcessor : public Task
{
 public:
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  void sendOutput(DataAllocator& output, long tf, const o2::parameters::GRPLHCIFData& lhcifdata);

  LHCIFfileReader mReader;
  bool mVerbose = false; // to enable verbose mode
};
} // namespace grp

namespace framework
{
DataProcessorSpec getGRPLHCIFfileSpec();

} // namespace framework
} // namespace o2

#endif
