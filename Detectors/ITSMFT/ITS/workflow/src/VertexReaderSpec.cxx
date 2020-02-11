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

#include "TTree.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "ITSWorkflow/VertexReaderSpec.h"

using namespace o2::framework;
using namespace o2::its;

namespace o2
{
namespace its
{

void VertexReader::init(InitContext& ic)
{
  mInputFileName = ic.options().get<std::string>("its-vertex-infile");
}

void VertexReader::run(ProcessingContext& pc)
{

  if (mFinished) {
    return;
  }

  // load data from files
  TFile trFile(mInputFileName.c_str(), "read");
  if (trFile.IsZombie()) {
    LOG(FATAL) << "Failed to open tracks file " << mInputFileName;
  }
  TTree* trTree = (TTree*)trFile.Get(mVertexTreeName.c_str());
  if (!trTree) {
    LOG(FATAL) << "Failed to load tracks tree " << mVertexTreeName << " from " << mInputFileName;
  }
  if (!trTree->GetBranch(mVertexBranchName.c_str())) {
    LOG(FATAL) << "No " << mVertexBranchName << " branch in " << mVertexTreeName;
  }
  if (!trTree->GetBranch(mVertexROFBranchName.c_str())) {
    LOG(FATAL) << "No " << mVertexROFBranchName << " branch in " << mVertexTreeName;
  }
  trTree->SetBranchAddress(mVertexBranchName.c_str(), &mVerticesInp);
  trTree->SetBranchAddress(mVertexROFBranchName.c_str(), &mVerticesROFRecInp);

  trTree->GetBranch(mVertexROFBranchName.c_str())->GetEntry(0);
  trTree->GetBranch(mVertexBranchName.c_str())->GetEntry(0);
  delete trTree;
  trFile.Close();

  LOG(INFO) << "ITSVertexReader pushes " << mVerticesROFRec.size() << " ROFRecords,"
            << mVertices.size() << " vertices";
  pc.outputs().snapshot(Output{"ITS", "VERTICES", 0, Lifetime::Timeframe}, mVertices);
  pc.outputs().snapshot(Output{"ITS", "VERTICESROF", 0, Lifetime::Timeframe}, mVerticesROFRec);

  mFinished = true;

  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
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
      {"its-vertex-infile", VariantType::String, "o2trac_its.root", {"Name of the input ITS vertex file"}}}};
}

} // namespace its
} // namespace o2
