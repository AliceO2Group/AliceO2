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
#include "FT0Workflow/ReconstructionSpec.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/MCLabel.h"
#include "FT0Calibration/FT0ChannelTimeCalibrationObject.h"
#include "Framework/CCDBParamSpec.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

void ReconstructionDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  LOG(info) << "ReconstructionDPL::init";
}

void ReconstructionDPL::run(ProcessingContext& pc)
{
  /*
  const auto ref = pc.inputs().getFirstValid(true);
  auto creationTime =
    o2::framework::DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(ref)->creation;
  auto& mCCDBManager = o2::ccdb::BasicCCDBManager::instance();
  mCCDBManager.setURL(mCCDBpath);
  mCCDBManager.setTimestamp(creationTime);
  LOG(debug) << " set-up CCDB " << mCCDBpath << " creationTime " << creationTime;
  */
  mTimer.Start(false);
  mRecPoints.clear();
  auto digits = pc.inputs().get<gsl::span<o2::ft0::Digit>>("digits");
  auto digch = pc.inputs().get<gsl::span<o2::ft0::ChannelData>>("digch");
  // RS: if we need to process MC truth, uncomment lines below
  //std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>> labels;
  //const o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>* lblPtr = nullptr;
  if (mUseMC) {
    LOG(info) << "Ignoring MC info";
  }
  if (mUpdateCCDB) {
    auto caliboffsets = pc.inputs().get<o2::ft0::FT0ChannelTimeCalibrationObject*>("ft0offsets");
    mReco.SetChannelOffset(caliboffsets.get());
    LOG(info) << "RecoSpec  mReco.SetChannelOffset(&caliboffsets)";
  }
  /*
  auto calibslew = mCCDBManager.get<std::array<TGraph, NCHANNELS>>("FT0/SlewingCorr");
  LOG(debug) << " calibslew " << calibslew;
  if (calibslew) {
    mReco.SetSlew(calibslew);
    LOG(info) << " calibslew set slew " << calibslew;
  }
  */
  int nDig = digits.size();
  LOG(debug) << " nDig " << nDig;
  mRecPoints.reserve(nDig);
  mRecChData.resize(digch.size());
  for (int id = 0; id < nDig; id++) {
    const auto& digit = digits[id];
    LOG(debug) << " ndig " << id << " bc " << digit.getBC() << " orbit " << digit.getOrbit();
    auto channels = digit.getBunchChannelData(digch);
    gsl::span<o2::ft0::ChannelDataFloat> out_ch(mRecChData);
    out_ch = out_ch.subspan(digit.ref.getFirstEntry(), digit.ref.getEntries());
    mRecPoints.emplace_back(mReco.process(digit, channels, out_ch));
  }
  // do we ignore MC in this task?

  LOG(debug) << "FT0 reconstruction pushes " << mRecPoints.size() << " RecPoints";
  pc.outputs().snapshot(Output{mOrigin, "RECPOINTS", 0, Lifetime::Timeframe}, mRecPoints);
  pc.outputs().snapshot(Output{mOrigin, "RECCHDATA", 0, Lifetime::Timeframe}, mRecChData);

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
  LOGF(info, "FT0 reconstruction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getReconstructionSpec(bool useMC, const std::string ccdbpath)
{
  std::vector<InputSpec> inputSpec;
  std::vector<OutputSpec> outputSpec;
  inputSpec.emplace_back("digits", o2::header::gDataOriginFT0, "DIGITSBC", 0, Lifetime::Timeframe);
  inputSpec.emplace_back("digch", o2::header::gDataOriginFT0, "DIGITSCH", 0, Lifetime::Timeframe);
  if (useMC) {
    LOG(info) << "Currently Reconstruction does not consume and provide MC truth";
    inputSpec.emplace_back("labels", o2::header::gDataOriginFT0, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  inputSpec.emplace_back("ft0offsets", "FT0", "TimeOffset", 0,
                         Lifetime::Condition,
                         ccdbParamSpec("FT0/Calib/ChannelTimeOffset"));

  outputSpec.emplace_back(o2::header::gDataOriginFT0, "RECPOINTS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back(o2::header::gDataOriginFT0, "RECCHDATA", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "ft0-reconstructor",
    inputSpec,
    outputSpec,
    AlgorithmSpec{adaptFromTask<ReconstructionDPL>(useMC, ccdbpath)},
    Options{}};
}

} // namespace ft0
} // namespace o2
