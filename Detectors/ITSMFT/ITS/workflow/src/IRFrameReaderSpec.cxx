// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "ITSWorkflow/IRFrameReaderSpec.h"
#include "CommonDataFormat/IRFrame.h"
#include "CommonUtils/StringUtils.h"
#include "TFile.h"
#include "TTree.h"

using namespace o2::framework;
using namespace o2::its;

namespace o2
{
namespace its
{

class IRFrameReaderSpec : public o2::framework::Task
{
 public:
  IRFrameReaderSpec() = default;
  ~IRFrameReaderSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);

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
                                                 ic.options().get<std::string>("its-irframe-infile"));
  connectTree(mInputFileName);
}

void IRFrameReaderSpec::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(INFO) << "Pushing " << mIRF.size() << " IR-frames in at entry " << ent;
  pc.outputs().snapshot(Output{"ITS", "IRFRAMES", 0, Lifetime::Timeframe}, mIRF);

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
  LOG(INFO) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getIRFrameReaderSpec()
{
  std::vector<OutputSpec> outputSpec;

  return DataProcessorSpec{
    "its-irframe-reader",
    Inputs{},
    Outputs{{"ITS", "IRFRAMES", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<IRFrameReaderSpec>()},
    Options{
      {"its-irframe-infile", VariantType::String, "o2_its_irframe.root", {"Name of the input IRFrames file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace its
} // namespace o2
