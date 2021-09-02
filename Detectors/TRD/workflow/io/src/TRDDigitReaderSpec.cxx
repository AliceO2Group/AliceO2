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

#include "TRDWorkflowIO/TRDDigitReaderSpec.h"


#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "CommonUtils/StringUtils.h"
#include "fairlogger/Logger.h"

#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include <SimulationDataFormat/IOMCTruthContainerView.h>

#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/TriggerRecord.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

void TRDDigitReaderSpec::init(InitContext& ic)
{

  auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                ic.options().get<std::string>("digitsfile"));

  mFile = std::make_unique<TFile>(filename.c_str(), "OLD");
  if (!(mFile && !mFile->IsZombie())) {
    throw std::runtime_error("Error opening TRD digits file");
  }
}

void TRDDigitReaderSpec::run(ProcessingContext& pc)
{
  auto DPLTree = ((TTree*)mFile->Get(mDigitTreeName.c_str()));
  if (DPLTree) {
    std::vector<o2::trd::Digit>* digits = nullptr;
    o2::dataformats::IOMCTruthContainerView* ioLabels = nullptr;
    std::vector<o2::trd::TriggerRecord>* triggerRecords = nullptr;

    auto getFromBranch = [DPLTree](const char* name, void** ptr) {
      auto br = DPLTree->GetBranch(name);
      br->SetAddress(ptr);
      br->GetEntry(0);
      br->ResetAddress();
    };
    getFromBranch(mDigitBranchName.c_str(), (void**)&digits);
    getFromBranch(mTriggerRecordBranchName.c_str(), (void**)&triggerRecords);
    if (mUseMC) {
      getFromBranch(mMCLabelsBranchName.c_str(), (void**)&ioLabels);
      // publish labels in shared memory
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{"TRD", "LABELS", 0, Lifetime::Timeframe});
      ioLabels->copyandflatten(sharedlabels);
      LOG(INFO) << "Labels size (in bytes) = " << sharedlabels.size();
    }

    pc.outputs().snapshot(Output{"TRD", "DIGITS", 0, Lifetime::Timeframe}, *digits);
    pc.outputs().snapshot(Output{"TRD", "TRGRDIG", 0, Lifetime::Timeframe}, *triggerRecords);
    LOG(INFO) << "Digits size=" << digits->size() << " triggerrecords size=" << triggerRecords->size();
  } else {
    LOG(ERROR) << "Error opening TTree";
  }

  mFile->Close();

  // send endOfData control event and mark the reader as ready to finish
  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getTRDDigitReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRD", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "TRGRDIG", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("TRD", "LABELS", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{"TRDDIGITREADER",
                           Inputs{},
                           outputs,
                           AlgorithmSpec{adaptFromTask<TRDDigitReaderSpec>(useMC)},
                           Options{
                             {"digitsfile", VariantType::String, "trddigits.root", {"Input data file containing run3 digitizer going into Trap Simulator"}},
                             {"input-dir", VariantType::String, "none", {"Input directory"}}}};
};

} //end namespace trd
} //end namespace o2
