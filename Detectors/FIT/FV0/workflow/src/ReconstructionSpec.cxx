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

/// @file   ReconstructionSpec.cxx

#include <vector>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/CCDBParamSpec.h"
#include "FV0Workflow/ReconstructionSpec.h"
#include "DataFormatsFV0/Digit.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/MCLabel.h"
#include "DataFormatsFV0/FV0ChannelTimeCalibrationObject.h"
#include "Framework/CCDBParamSpec.h"

using namespace o2::framework;

namespace o2
{
namespace fv0
{

void ReconstructionDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  LOG(info) << "ReconstructionDPL::init";
}

void ReconstructionDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  mRecPoints.clear();
  auto digits = pc.inputs().get<gsl::span<o2::fv0::Digit>>("digits");
  auto digch = pc.inputs().get<gsl::span<o2::fv0::ChannelData>>("digch");
  // RS: if we need to process MC truth, uncomment lines below
  // std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>> labels;
  // const o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>* lblPtr = nullptr;
  if (mUseMC) {
    LOG(info) << "Ignoring MC info";
  }
  if (mUpdateCCDB) {
    auto caliboffsets = pc.inputs().get<o2::fv0::FV0ChannelTimeCalibrationObject*>("fv0offsets");
    mReco.SetChannelOffset(caliboffsets.get());
  }

  int nDig = digits.size();
  LOG(debug) << " nDig " << nDig << " | ndigch " << digch.size();
  mRecPoints.reserve(nDig);
  mRecChData.resize(digch.size());
  for (int id = 0; id < nDig; id++) {
    const auto& digit = digits[id];
    LOG(debug) << " ndig " << id << " bc " << digit.getBC() << " orbit " << digit.getOrbit();
    auto channels = digit.getBunchChannelData(digch);
    gsl::span<o2::fv0::ChannelDataFloat> out_ch(mRecChData);
    out_ch = out_ch.subspan(digit.ref.getFirstEntry(), digit.ref.getEntries());
    mRecPoints.emplace_back(mReco.process(digit, channels, out_ch));
  }

  LOG(debug) << "FV0 reconstruction pushes " << mRecPoints.size() << " RecPoints";
  pc.outputs().snapshot(Output{mOrigin, "RECPOINTS", 0}, mRecPoints);
  pc.outputs().snapshot(Output{mOrigin, "RECCHDATA", 0}, mRecChData);

  mTimer.Stop();
}
//_______________________________________
void ReconstructionDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("FT0", "TimeOffset", 0)) {
    mUpdateCCDB = false;
    return;
  }
}

void ReconstructionDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "FV0 reconstruction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getReconstructionSpec(bool useMC, const std::string ccdbpath)
{
  std::vector<InputSpec> inputSpec;
  std::vector<OutputSpec> outputSpec;
  inputSpec.emplace_back("digits", o2::header::gDataOriginFV0, "DIGITSBC", 0, Lifetime::Timeframe);
  inputSpec.emplace_back("digch", o2::header::gDataOriginFV0, "DIGITSCH", 0, Lifetime::Timeframe);
  if (useMC) {
    LOG(info) << "Currently Reconstruction does not consume and provide MC truth";
    inputSpec.emplace_back("labels", o2::header::gDataOriginFV0, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  inputSpec.emplace_back("fv0offsets", "FV0", "TimeOffset", 0,
                         Lifetime::Condition,
                         ccdbParamSpec("FV0/Calib/ChannelTimeOffset"));

  outputSpec.emplace_back(o2::header::gDataOriginFV0, "RECPOINTS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back(o2::header::gDataOriginFV0, "RECCHDATA", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "fv0-reconstructor",
    inputSpec,
    outputSpec,
    AlgorithmSpec{adaptFromTask<ReconstructionDPL>(useMC, ccdbpath)},
    Options{}};
}

} // namespace fv0
} // namespace o2
