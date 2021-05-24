// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   VertexReaderSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "ITSWorkflow/VertexReaderSpec.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;
using namespace o2::its;

namespace o2
{
namespace its
{

void VertexReader::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("its-vertex-infile"));
  connectTree(mFileName);
}

void VertexReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(INFO) << "Pushing " << mVerticesPtr->size() << " vertices in " << mVerticesROFRecPtr->size()
            << " ROFs at entry " << ent;
  pc.outputs().snapshot(Output{"ITS", "VERTICES", 0, Lifetime::Timeframe}, mVertices);
  pc.outputs().snapshot(Output{"ITS", "VERTICESROF", 0, Lifetime::Timeframe}, mVerticesROFRec);

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void VertexReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mVertexTreeName.c_str()));
  assert(mTree);
  assert(mTree->GetBranch(mVertexBranchName.c_str()));
  assert(mTree->GetBranch(mVertexROFBranchName.c_str()));
  mTree->SetBranchAddress(mVertexBranchName.c_str(), &mVerticesPtr);
  mTree->SetBranchAddress(mVertexROFBranchName.c_str(), &mVerticesROFRecPtr);
  LOG(INFO) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getITSVertexReaderSpec()
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("ITS", "VERTICES", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "VERTICESROF", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "its-vertex-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<VertexReader>()},
    Options{
      {"its-vertex-infile", VariantType::String, "o2trac_its.root", {"Name of the input ITS vertex file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace its
} // namespace o2
