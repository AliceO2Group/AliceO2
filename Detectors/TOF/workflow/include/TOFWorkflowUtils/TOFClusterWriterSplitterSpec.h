// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TOFCLUSTER_SPLITTER_WRITER_H
#define O2_TOFCLUSTER_SPLITTER_WRITER_H

/// @file   TOFClusterWriterSplitterSpec.h
/// @brief  Device to write to tree the information for TOF time slewing calibration.

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsTOF/Cluster.h"
#include "Framework/Logger.h"
#include <TTree.h>
#include <TFile.h>
#include <gsl/span>

using namespace o2::framework;

namespace o2
{
namespace tof
{
class TOFClusterWriterSplitter : public Task
{
  using OutputType = std::vector<o2::tof::Cluster>;

  std::string mBaseName;

 public:
  TOFClusterWriterSplitter(int nTF) : mTFthr(nTF) {}

  void createAndOpenFileAndTree()
  {
    TString filename = TString::Format("%s_%06d.root", mBaseName.c_str(), mCount);
    LOG(DEBUG) << "opening file " << filename.Data();
    mfileOut.reset(TFile::Open(TString::Format("%s", filename.Data()), "RECREATE"));
    mOutputTree = std::make_unique<TTree>("o2sim", "Tree with TOF clusters");
    mOutputTree->Branch("TOFCluster", &mPClusters);

    mNTF = 0;
  }

  void init(o2::framework::InitContext& ic) final
  {
    mBaseName = ic.options().get<std::string>("output-base-name");

    mCount = 0;
    createAndOpenFileAndTree();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto clusters = pc.inputs().get<OutputType>("clusters");
    mPClusters = &clusters;
    mOutputTree->Fill();

    mNTF++;

    if (mNTF >= mTFthr) {
      sendOutput();
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    mIsEndOfStream = true;
    sendOutput();
  }

 private:
  int mCount = 0; // how many times we filled the tree
  int mNTF = 0;
  int mTFthr = 1;
  bool mIsEndOfStream = false;
  OutputType mClusters;
  const OutputType* mPClusters = &mClusters;

  std::unique_ptr<TTree> mOutputTree;        ///< tree for the collected calib tof info
  std::unique_ptr<TFile> mfileOut = nullptr; // file in which to write the output

  //________________________________________________________________
  void sendOutput()
  {
    // This is to fill the tree.
    // One file with an empty tree will be created at the end, because we have to have a
    // tree opened before processing, since we do not know a priori if something else
    // will still come. The size of this extra file is ~6.5 kB

    mfileOut->cd();
    mOutputTree->Write();
    mOutputTree.reset();
    mfileOut.reset();
    mCount++;
    if (!mIsEndOfStream) {
      createAndOpenFileAndTree();
    }
  }
};
} // namespace tof

namespace framework
{

DataProcessorSpec getTOFClusterWriterSplitterSpec(int nTF)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("clusters", o2::header::gDataOriginTOF, "CLUSTERS");

  std::vector<OutputSpec> outputs; // empty

  return DataProcessorSpec{
    "tof-cluster-splitter-writer",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::tof::TOFClusterWriterSplitter>(nTF)},
    Options{{"output-base-name", VariantType::String, "tofclusters", {"Name of the input file (root extension will be added)"}}}};
}

} // namespace framework
} // namespace o2

#endif
