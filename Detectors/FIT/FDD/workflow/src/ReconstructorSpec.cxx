// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ReconstructorSpec.cxx

#include <vector>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Framework/ControlService.h"
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
  if (mFinished) {
    return;
  }
  mRecPoints.clear();
  auto digits = pc.inputs().get<const std::vector<o2::fdd::Digit>>("digits");
  // RS: if we need to process MC truth, uncomment lines below
  //std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>> labels;
  //const o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>* lblPtr = nullptr;
  if (mUseMC) {
    //labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>*>("labels");
    //lblPtr = labels.get();
    LOG(INFO) << "Ignoring MC info";
  }
  int nDig = digits.size();
  mRecPoints.resize(nDig);
  for (int id = 0; id < nDig; id++) {
    const auto& digit = digits[id];
    auto& rp = mRecPoints[id];
    mReco.Process(digit, rp);
  }
  // do we ignore MC in this task?

  LOG(INFO) << "FDD reconstruction pushes " << mRecPoints.size() << " RecPoints";
  pc.outputs().snapshot(Output{mOrigin, "RECPOINTS", 0, Lifetime::Timeframe}, mRecPoints);

  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(false);
}

DataProcessorSpec getFDDReconstructorSpec(bool useMC)
{
  std::vector<InputSpec> inputSpec;
  std::vector<OutputSpec> outputSpec;
  inputSpec.emplace_back("digits", o2::header::gDataOriginFDD, "DIGITS", 0, Lifetime::Timeframe);
  if (useMC) {
    LOG(INFO) << "Currently FDDReconstructor does not consume and provide MC truth";
    // inputSpec.emplace_back("labels", o2::header::gDataOriginFDD, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }
  outputSpec.emplace_back(o2::header::gDataOriginFDD, "RECPOINTS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "fdd-reconstructor",
    inputSpec,
    outputSpec,
    AlgorithmSpec{adaptFromTask<FDDReconstructorDPL>(useMC)},
    Options{}};
}

} // namespace fdd
} // namespace o2
