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
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "FDDWorkflow/RecPointReaderSpec.h"
#include "DetectorsCommonDataFormats/NameConf.h"

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
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("fdd-recpoints-infile"));
  connectTree(mInputFileName);
}

void RecPointReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);

  LOG(INFO) << "FDD RecPointReader pushes " << mRecPoints->size() << " recpoints at entry " << ent;
  pc.outputs().snapshot(Output{mOrigin, "RECPOINTS", 0, Lifetime::Timeframe}, *mRecPoints);

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void RecPointReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mRecPointTreeName.c_str()));
  assert(mTree);

  mTree->SetBranchAddress(mRecPointBranchName.c_str(), &mRecPoints);
  if (mUseMC) {
    LOG(WARNING) << "MC-truth is not supported for FDD recpoints currently";
    mUseMC = false;
  }

  LOG(INFO) << "Loaded FDD RecPoints tree from " << filename << " with " << mTree->GetEntries() << " entries";
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
      {"fdd-recpoints-infile", VariantType::String, "o2reco_fdd.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace fdd
} // namespace o2
