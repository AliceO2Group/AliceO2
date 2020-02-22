// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0ReconstructorSpec.cxx

#include <vector>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Framework/ControlService.h"
#include "FITWorkflow/FT0ReconstructorSpec.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/MCLabel.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

void FT0ReconstructorDPL::init(InitContext& ic)
{
  LOG(INFO) << "FT0ReconstructorDPL::init";
}

void FT0ReconstructorDPL::run(ProcessingContext& pc)
{
  if (mFinished) {
    return;
  }
  mRecPoints.clear();
  auto digits = pc.inputs().get<gsl::span<o2::ft0::Digit>>("digits");
  //auto digits = pc.inputs().get<const std::vector<o2::ft0::Digit>>("digits");
  auto digch = pc.inputs().get<gsl::span<o2::ft0::ChannelData>>("digch");
  // RS: if we need to process MC truth, uncomment lines below
  //std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>> labels;
  //const o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>* lblPtr = nullptr;
  if (mUseMC) {
    //   labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>*>("labels");
    // lblPtr = labels.get();
    LOG(INFO) << "Ignoring MC info";
  }
  int nDig = digits.size();
  mRecPoints.reserve(nDig);
  mRecChData.resize(digch.size());
  for (int id = 0; id < nDig; id++) {
    const auto& digit = digits[id];
    auto channels = digit.getBunchChannelData(digch);
    gsl::span<o2::ft0::ChannelDataFloat> out_ch(mRecChData);
    out_ch = out_ch.subspan(digit.ref.getFirstEntry(), digit.ref.getEntries());
    mRecPoints.emplace_back(mReco.process(digit, channels, out_ch));
  }
  // do we ignore MC in this task?

  LOG(DEBUG) << "FT0 reconstruction pushes " << mRecPoints.size() << " RecPoints";
  pc.outputs().snapshot(Output{mOrigin, "RECPOINTS", 0, Lifetime::Timeframe}, mRecPoints);
  pc.outputs().snapshot(Output{mOrigin, "RECCHDATA", 0, Lifetime::Timeframe}, mRecChData);

  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getFT0ReconstructorSpec(bool useMC)
{
  std::vector<InputSpec> inputSpec;
  std::vector<OutputSpec> outputSpec;
  inputSpec.emplace_back("digits", o2::header::gDataOriginFT0, "DIGITSBC", 0, Lifetime::Timeframe);
  inputSpec.emplace_back("digch", o2::header::gDataOriginFT0, "DIGITSCH", 0, Lifetime::Timeframe);
  if (useMC) {
    LOG(INFO) << "Currently FT0Reconstructor does not consume and provide MC truth";
    inputSpec.emplace_back("labels", o2::header::gDataOriginFT0, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  outputSpec.emplace_back(o2::header::gDataOriginFT0, "RECPOINTS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back(o2::header::gDataOriginFT0, "RECCHDATA", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "ft0-reconstructor",
    inputSpec,
    outputSpec,
    AlgorithmSpec{adaptFromTask<FT0ReconstructorDPL>(useMC)},
    Options{}};
}

} // namespace ft0
} // namespace o2
