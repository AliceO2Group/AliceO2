// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0RecPointReaderSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "FITWorkflow/FT0RecPointReaderSpec.h"

using namespace o2::framework;
using namespace o2::ft0;

namespace o2
{
namespace ft0
{

RecPointReader::RecPointReader(bool useMC)
{
  mUseMC = useMC;
  if (useMC) {
    LOG(WARNING) << "FT0 RecPoint reader at the moment does not process MC";
  }
}

void RecPointReader::init(InitContext& ic)
{
  mInputFileName = ic.options().get<std::string>("ft0-recpoints-infile");
}

void RecPointReader::run(ProcessingContext& pc)
{
  if (mFinished) {
    return;
  }

  { // load data from files
    TFile rpFile(mInputFileName.c_str(), "read");
    if (rpFile.IsZombie()) {
      LOG(FATAL) << "Failed to open FT0 recpoints file " << mInputFileName;
    }
    TTree* rpTree = (TTree*)rpFile.Get(mRecPointTreeName.c_str());
    if (!rpTree) {
      LOG(FATAL) << "Failed to load FT0 recpoints tree " << mRecPointTreeName << " from " << mInputFileName;
    }
    LOG(INFO) << "Loaded FT0 recpoints tree " << mRecPointTreeName << " from " << mInputFileName;

    rpTree->SetBranchAddress(mRecPointBranchName.c_str(), &mRecPoints);
    if (mUseMC) {
      LOG(WARNING) << "MC-truth is not supported for FT0 recpoints currently";
      mUseMC = false;
    }

    rpTree->GetEntry(0);
    delete rpTree;
    rpFile.Close();
  }

  LOG(INFO) << "FT0 RecPointReader pushes " << mRecPoints->size() << " recpoints";
  pc.outputs().snapshot(Output{mOrigin, "RECPOINTS", 0, Lifetime::Timeframe}, *mRecPoints);

  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getFT0RecPointReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back(o2::header::gDataOriginFT0, "RECPOINTS", 0, Lifetime::Timeframe);
  if (useMC) {
    LOG(WARNING) << "MC-truth is not supported for FT0 recpoints currently";
  }

  return DataProcessorSpec{
    "ft0-recpoints-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<RecPointReader>()},
    Options{
      {"ft0-recpoints-infile", VariantType::String, "o2reco_ft0.root", {"Name of the input file"}}}};
}

} // namespace ft0
} // namespace o2
