// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_INTERPOLATION_SPEC_H
#define O2_TPC_INTERPOLATION_SPEC_H

/// @file   TPCInterpolationSpec.h

#include "DataFormatsTPC/Constants.h"
#include "SpacePoints/TrackInterpolation.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TStopwatch.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{
class TPCInterpolationDPL : public Task
{
 public:
  TPCInterpolationDPL(bool useMC) : mUseMC(useMC) {}
  ~TPCInterpolationDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  o2::tpc::TrackInterpolation mInterpolation; // track interpolation engine
  bool mUseMC{false}; ///< MC flag
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getTPCInterpolationSpec(bool useMC);

} // namespace tpc
} // namespace o2

#endif
