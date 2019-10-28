// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  RecPointReaderSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "FDDWorkflow/RecPointReaderSpec.h"

using namespace o2::framework;
using namespace o2::fdd;

namespace o2
{
namespace fdd
{

RecPointReader::RecPointReader(bool useMC)
{
  mUseMC = useMC;
  if (useMC) {
    LOG(WARNING) << "FDD RecPoint reader at the moment does not process MC";
  }
}

void RecPointReader::init(InitContext& ic)
{
  mInputFileName = ic.options().get<std::string>("fdd-recpoints-infile");
}

void RecPointReader::run(ProcessingContext& pc)
{
  if (mFinished) {
    return;
  }

  { // load data from files
    TFile rpFile(mInputFileName.c_str(), "read");
    if (rpFile.IsZombie()) {
      LOG(FATAL) << "Failed to open FDD recpoints file " << mInputFileName;
    }
    TTree* rpTree = (TTree*)rpFile.Get(mRecPointTreeName.c_str());
    if (!rpTree) {
      LOG(FATAL) << "Failed to load FDD recpoints tree " << mRecPointTreeName << " from " << mInputFileName;
    }
    LOG(INFO) << "Loaded FDD recpoints tree " << mRecPointTreeName << " from " << mInputFileName;

    rpTree->SetBranchAddress(mRecPointBranchName.c_str(), &mRecPoints);
    if (mUseMC) {
      LOG(WARNING) << "MC-truth is not supported for FDD recpoints currently";
      mUseMC = false;
    }

    rpTree->GetEntry(0);
    delete rpTree;
    rpFile.Close();
  }

  LOG(INFO) << "FDD RecPointReader pushes " << mRecPoints->size() << " recpoints";
  pc.outputs().snapshot(Output{mOrigin, "RECPOINTS", 0, Lifetime::Timeframe}, *mRecPoints);

  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getFDDRecPointReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back(o2::header::gDataOriginFDD, "RECPOINTS", 0, Lifetime::Timeframe);
  if (useMC) {
    LOG(WARNING) << "MC-truth is not supported for FDD recpoints currently";
  }

  return DataProcessorSpec{
    "fdd-recpoints-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<RecPointReader>()},
    Options{
      {"fdd-recpoints-infile", VariantType::String, "o2reco_fdd.root", {"Name of the input file"}}}};
}

} // namespace fdd
} // namespace o2
