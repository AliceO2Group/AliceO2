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

/// @file   IRFrameReaderSpec.cxx

#include <vector>
#include <cassert>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "GlobalTrackingWorkflowReaders/IRFrameReaderSpec.h"
#include "CommonDataFormat/IRFrame.h"
#include "CommonUtils/StringUtils.h"
#include "TFile.h"
#include "TTree.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

class IRFrameReaderSpec : public o2::framework::Task
{
 public:
  IRFrameReaderSpec(o2::header::DataOrigin origin, uint32_t subSpec) : mDataOrigin(origin), mSubSpec(subSpec) {}
  ~IRFrameReaderSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);
  o2::header::DataOrigin mDataOrigin = o2::header::gDataOriginInvalid;
  uint32_t mSubSpec = 0;
  std::vector<o2::dataformats::IRFrame> mIRF, *mIRFInp = &mIRF;
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mInputFileName = "";
  std::string mTreeName = "o2sim";
  std::string mBranchName = "IRFrames";
};

void IRFrameReaderSpec::init(InitContext& ic)
{
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("irframe-infile"));
  connectTree(mInputFileName);
}

void IRFrameReaderSpec::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(debug) << "Pushing " << mIRF.size() << " IR-frames in at entry " << ent;
  pc.outputs().snapshot(Output{mDataOrigin, "IRFRAMES", mSubSpec}, mIRF);

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void IRFrameReaderSpec::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mTreeName.c_str()));
  assert(mTree);
  assert(mTree->GetBranch(mBranchName.c_str()));

  mTree->SetBranchAddress(mBranchName.c_str(), &mIRFInp);
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getIRFrameReaderSpec(o2::header::DataOrigin origin, uint32_t subSpec, const std::string& devName, const std::string& defFileName)
{
  std::vector<OutputSpec> outputSpec;

  return DataProcessorSpec{
    devName,
    Inputs{},
    Outputs{{origin, "IRFRAMES", subSpec, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<IRFrameReaderSpec>(origin, subSpec)},
    Options{
      {"irframe-infile", VariantType::String, defFileName, {"Name of the input IRFrames file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace globaltracking
} // namespace o2
