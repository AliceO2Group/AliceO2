// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


/// @file
/// @brief


#include <Framework/ConfigContext.h>
#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "TTree.h"
#include "TFile.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"

using namespace o2::framework;

namespace o2::calibration::fit
{

class FT0DigitsProducer final: public Task
{
 public:
  void init(InitContext& ic) final
  {
    auto filename = ic.options().get<std::string>("ft0-input-digit-file");
    mFile = std::make_unique<TFile>(filename.c_str(), "OLD");
    if (!mFile->IsOpen()) {
      LOG(ERROR) << "Cannot open the " << filename.c_str() << " file!";
      throw std::runtime_error("Cannot open input digits file");
    }
    mTree.reset((TTree*)mFile->Get("o2sim"));
    if (!mTree) {
      LOG(ERROR) << "Invalid digits file: " << filename.c_str();
      throw std::runtime_error("Invalid digits file");
    }
  }

  void run(ProcessingContext& pc) final
  {
    std::vector<o2::ft0::Digit> digits, *pDigits = &digits;
    std::vector<o2::ft0::ChannelData> channels, *pChannels = &channels;
    mTree->SetBranchAddress("FT0DIGITSBC", &pDigits);
    mTree->SetBranchAddress("FT0DIGITSCH", &pChannels);

    if (mCounterTF >= mTree->GetEntries()) {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }

    mTree->GetEntry(mCounterTF);
    pc.outputs().snapshot(Output{"FT0", "DIGITSBC", 0, Lifetime::Timeframe}, digits);
    pc.outputs().snapshot(Output{"FT0", "DIGITSCH", 0, Lifetime::Timeframe}, channels);
    ++mCounterTF;
  }


private:
  unsigned int mCounterTF = 0;
  std::unique_ptr<TTree> mTree;
  std::unique_ptr<TFile> mFile;

};

} // namespace o2

#include "Framework/runDataProcessing.h"


WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  WorkflowSpec  workflow;
  DataProcessorSpec spec
  {
    "FT0DigitsProducer",
      Inputs{},
      Outputs{
         {{"channels"}, "FT0", "DIGITSCH"},
         {{"digits"}, "FT0", "DIGITSBC"}},
      AlgorithmSpec{  adaptFromTask<o2::calibration::fit::FT0DigitsProducer>()},
      Options{
        {"ft0-input-digit-file", VariantType::String, "ft0digits.root", { "path to digits file" }}
      }
  };

  workflow.emplace_back(spec);
  return workflow;
}

