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

/// @file   ReconstructorSpec.cxx

#include <vector>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "FDDWorkflow/ReconstructorSpec.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/MCLabel.h"

using namespace o2::framework;

namespace o2
{
namespace fdd
{

void FDDReconstructorDPL::init(InitContext& ic)
{
}

void FDDReconstructorDPL::run(ProcessingContext& pc)
{
  mRecPoints.clear();
  mRecChData.clear();
  auto digitsBC = pc.inputs().get<gsl::span<o2::fdd::Digit>>("digitsBC");
  auto digitsCh = pc.inputs().get<gsl::span<o2::fdd::ChannelData>>("digitsCh");
  // RS: if we need to process MC truth, uncomment lines below
  // std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>> labels;
  // const o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>* lblPtr = nullptr;
  if (mUseMC) {
    // labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>*>("labels");
    // lblPtr = labels.get();
    LOG(info) << "Ignoring MC info";
  }
  int nDig = digitsBC.size();
  mRecPoints.reserve(nDig);
  mRecChData.reserve(digitsCh.size());
  for (int id = 0; id < nDig; id++) {
    const auto& digit = digitsBC[id];
    auto channels = digit.getBunchChannelData(digitsCh);
    mReco.process(digit, channels, mRecPoints, mRecChData);
  }
  // do we ignore MC in this task?
  LOG(debug) << "FDD reconstruction pushes " << mRecPoints.size() << " RecPoints";
  pc.outputs().snapshot(Output{mOrigin, "RECPOINTS", 0, Lifetime::Timeframe}, mRecPoints);
  pc.outputs().snapshot(Output{mOrigin, "RECCHDATA", 0, Lifetime::Timeframe}, mRecChData);
}

DataProcessorSpec getFDDReconstructorSpec(bool useMC)
{
  std::vector<InputSpec> inputSpec;
  std::vector<OutputSpec> outputSpec;
  inputSpec.emplace_back("digitsBC", o2::header::gDataOriginFDD, "DIGITSBC", 0, Lifetime::Timeframe);
  inputSpec.emplace_back("digitsCh", o2::header::gDataOriginFDD, "DIGITSCH", 0, Lifetime::Timeframe);
  if (useMC) {
    LOG(info) << "Currently FDDReconstructor does not consume and provide MC truth";
    // inputSpec.emplace_back("labels", o2::header::gDataOriginFDD, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  outputSpec.emplace_back(o2::header::gDataOriginFDD, "RECPOINTS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back(o2::header::gDataOriginFDD, "RECCHDATA", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "fdd-reconstructor",
    inputSpec,
    outputSpec,
    AlgorithmSpec{adaptFromTask<FDDReconstructorDPL>(useMC)},
    Options{}};
}

} // namespace fdd
} // namespace o2
