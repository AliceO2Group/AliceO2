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
#include "DataFormatsFT0/DigitFilterParam.h"
#include "DataFormatsFT0/CalibParam.h"
#include "DataFormatsFT0/MCLabel.h"
#include "DataFormatsFT0/SpectraInfoObject.h"
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
  o2::ft0::ChannelFilterParam::Instance().printKeyValues();
  o2::ft0::TimeFilterParam::Instance().printKeyValues();
  // Parameters which are used in reco, too many will be printed if use printKeyValues()
  LOG(info) << "FT0 param mMinEntriesThreshold: " << CalibParam::Instance().mMinEntriesThreshold;
  LOG(info) << "FT0 param mMaxEntriesThreshold:" << CalibParam::Instance().mMaxEntriesThreshold;
  LOG(info) << "FT0 param mMinRMS: " << CalibParam::Instance().mMinRMS;
  LOG(info) << "FT0 param mMaxSigma: " << CalibParam::Instance().mMaxSigma;
  LOG(info) << "FT0 param mMaxDiffMean: " << CalibParam::Instance().mMaxDiffMean;
}

void ReconstructionDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  mRecPoints.clear();
  mRecChData.clear();
  auto digits = pc.inputs().get<gsl::span<o2::ft0::Digit>>("digits");
  auto channels = pc.inputs().get<gsl::span<o2::ft0::ChannelData>>("digch");
  // RS: if we need to process MC truth, uncomment lines below
  // std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>> labels;
  // const o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>* lblPtr = nullptr;
  if (mUseMC) {
    LOG(info) << "Ignoring MC info";
  }
  if (mUpdateCCDB) {
    auto timeCalibObject = pc.inputs().get<o2::ft0::TimeSpectraInfoObject*>("ft0_timespectra");
    mReco.SetTimeCalibObject(timeCalibObject.get());
  }
  /*
  auto calibslew = mCCDBManager.get<std::array<TGraph, NCHANNELS>>("FT0/SlewingCorr");
  LOG(debug) << " calibslew " << calibslew;
  if (calibslew) {
    mReco.SetSlew(calibslew);
    LOG(info) << " calibslew set slew " << calibslew;
  }
  */
  mRecPoints.reserve(digits.size());
  mRecChData.reserve(channels.size());
  mReco.processTF(digits, channels, mRecPoints, mRecChData);
  // do we ignore MC in this task?
  LOG(debug) << "FT0 reconstruction pushes " << mRecPoints.size() << " RecPoints";
  pc.outputs().snapshot(Output{mOrigin, "RECPOINTS", 0, Lifetime::Timeframe}, mRecPoints);
  pc.outputs().snapshot(Output{mOrigin, "RECCHDATA", 0, Lifetime::Timeframe}, mRecChData);

  mTimer.Stop();
}
//_______________________________________
void ReconstructionDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("FT0", "TimeSpectraInfo", 0)) {
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
  inputSpec.emplace_back("ft0_timespectra", "FT0", "TimeSpectraInfo", 0,
                         Lifetime::Condition,
                         ccdbParamSpec("FT0/Calib/TimeSpectraInfo"));

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
