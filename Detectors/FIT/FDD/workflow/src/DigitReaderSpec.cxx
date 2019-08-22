// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitReaderSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "FDDWorkflow/DigitReaderSpec.h"

using namespace o2::framework;
using namespace o2::fdd;

namespace o2
{
namespace fdd
{

DigitReader::DigitReader(bool useMC)
{
  mUseMC = useMC;
}

void DigitReader::init(InitContext& ic)
{
  mInputFileName = ic.options().get<std::string>("fdd-digits-infile");
}

void DigitReader::run(ProcessingContext& pc)
{
  if (mFinished) {
    return;
  }

  { // load data from files
    TFile digFile(mInputFileName.c_str(), "read");
    if (digFile.IsZombie()) {
      LOG(FATAL) << "Failed to open FDD digits file " << mInputFileName;
    }
    TTree* digTree = (TTree*)digFile.Get(mDigitTreeName.c_str());
    if (!digTree) {
      LOG(FATAL) << "Failed to load FDD digits tree " << mDigitTreeName << " from " << mInputFileName;
    }
    LOG(INFO) << "Loaded FDD digits tree " << mDigitTreeName << " from " << mInputFileName;

    digTree->SetBranchAddress(mDigitBranchName.c_str(), &mDigits);
    if (mUseMC) {
      if (digTree->GetBranch(mDigitMCTruthBranchName.c_str())) {
        digTree->SetBranchAddress(mDigitMCTruthBranchName.c_str(), &mMCTruth);
        LOG(INFO) << "Will use MC-truth from " << mDigitMCTruthBranchName;
      } else {
        LOG(INFO) << "MC-truth is missing";
        mUseMC = false;
      }
    }
    digTree->GetEntry(0);
    delete digTree;
    digFile.Close();
  }

  LOG(INFO) << "FDD DigitReader pushes " << mDigits->size() << " digits";
  pc.outputs().snapshot(Output{mOrigin, "DIGITS", 0, Lifetime::Timeframe}, *mDigits);
  if (mUseMC) {
    pc.outputs().snapshot(Output{mOrigin, "DIGITSMCTR", 0, Lifetime::Timeframe}, *mMCTruth);
  }

  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(false);
}

DataProcessorSpec getFDDDigitReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back(o2::header::gDataOriginFDD, "DIGITS", 0, Lifetime::Timeframe);
  if (useMC) {
    outputSpec.emplace_back(o2::header::gDataOriginFDD, "DIGITSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "fdd-digit-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<DigitReader>()},
    Options{
      {"fdd-digits-infile", VariantType::String, "fdddigits.root", {"Name of the input file"}}}};
}

} // namespace fdd
} // namespace o2
