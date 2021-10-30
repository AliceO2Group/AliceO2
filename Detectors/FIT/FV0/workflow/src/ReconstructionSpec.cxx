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
#include "FV0Workflow/ReconstructionSpec.h"
#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/MCLabel.h"
#include "FV0Calibration/FV0ChannelTimeCalibrationObject.h"

using namespace o2::framework;

namespace o2
{
namespace fv0
{

void ReconstructionDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  LOG(INFO) << "ReconstructionDPL::init";
}

void ReconstructionDPL::run(ProcessingContext& pc)
{
  auto& mCCDBManager = o2::ccdb::BasicCCDBManager::instance();
  //  mCCDBManager.setURL("http://ccdb-test.cern.ch:8080");
  mCCDBManager.setURL(mCCDBpath);
  LOG(debug) << " set-up CCDB " << mCCDBpath;
  mTimer.Start(false);
  mRecPoints.clear();
  auto digits = pc.inputs().get<gsl::span<o2::fv0::BCData>>("digits");
  auto digch = pc.inputs().get<gsl::span<o2::fv0::ChannelData>>("digch");
  // RS: if we need to process MC truth, uncomment lines below
  //std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>> labels;
  //const o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>* lblPtr = nullptr;
  if (mUseMC) {
    //   labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>*>("labels");
    // lblPtr = labels.get();
    LOG(debug) << "Ignoring MC info";
  }
  auto caliboffsets = mCCDBManager.get<o2::fv0::FV0ChannelTimeCalibrationObject>("FV0/Calibration/ChannelTimeOffset");
  mReco.setChannelOffset(caliboffsets);
  int nDig = digits.size();
  LOG(debug) << " nDig " << nDig << " | ndigch " << digch.size();
  mRecPoints.reserve(nDig);
  mRecChData.resize(digch.size());
  for (int id = 0; id < nDig; id++) {
    const auto& digit = digits[id];
    LOG(debug) << " ndig " << id << " bc " << digit.getIntRecord().bc << " orbit " << digit.getIntRecord().orbit;
    auto channels = digit.getBunchChannelData(digch);
    gsl::span<o2::fv0::ChannelDataFloat> out_ch(mRecChData);
    out_ch = out_ch.subspan(digit.ref.getFirstEntry(), digit.ref.getEntries());
    mRecPoints.emplace_back(mReco.process(digit, channels, out_ch));
  }

  LOG(DEBUG) << "FV0 reconstruction pushes " << mRecPoints.size() << " RecPoints";
  pc.outputs().snapshot(Output{mOrigin, "RECPOINTS", 0, Lifetime::Timeframe}, mRecPoints);
  pc.outputs().snapshot(Output{mOrigin, "RECCHDATA", 0, Lifetime::Timeframe}, mRecChData);

  mTimer.Stop();
}

void ReconstructionDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "FV0 reconstruction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getReconstructionSpec(bool useMC, const std::string ccdbpath)
{
  std::vector<InputSpec> inputSpec;
  std::vector<OutputSpec> outputSpec;
  inputSpec.emplace_back("digits", o2::header::gDataOriginFV0, "DIGITSBC", 0, Lifetime::Timeframe);
  inputSpec.emplace_back("digch", o2::header::gDataOriginFV0, "DIGITSCH", 0, Lifetime::Timeframe);
  if (useMC) {
    LOG(INFO) << "Currently Reconstruction does not consume and provide MC truth";
    inputSpec.emplace_back("labels", o2::header::gDataOriginFV0, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
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
