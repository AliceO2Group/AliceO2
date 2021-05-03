// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDWorkflowIO/TRDDigitReaderSpec.h"

// this is somewhat assuming that a DPL workflow will run on one node

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
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
  LOG(info) << "input filename is :  " << ic.options().get<std::string>("digitsfile").c_str();

  mInputFileName = ic.options().get<std::string>("digitsfile");
  mFile = std::make_unique<TFile>(mInputFileName.c_str(), "OLD");
  if (!mFile->IsOpen()) {
    LOG(error) << "Cannot open digits input file : " << mInputFileName;
    mState = 0; //prevent from getting into run method.

  } else {
    mState = 1;
  }
}

void TRDDigitReaderSpec::run(ProcessingContext& pc)
{
  if (mState != 1) {
    LOG(info) << "mState is not 1";
    return;
  }
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
      LOG(info) << "TRDDigitReader labels size (in bytes) = " << sharedlabels.size();
    }

    pc.outputs().snapshot(Output{"TRD", "DIGITS", 0, Lifetime::Timeframe}, *digits);
    pc.outputs().snapshot(Output{"TRD", "TRGRDIG", 0, Lifetime::Timeframe}, *triggerRecords);
    LOG(info) << "TRDDigitReader digits size=" << digits->size() << " triggerrecords size=" << triggerRecords->size();
  }
  //delete DPLTree; // next line will delete the pointer as well.
  mFile->Close();

  mState = 2; // prevent coming in here again.
              // send endOfData control event and mark the reader as ready to finish
  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getTRDDigitReaderSpec(int channels, bool useMC)
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
                           AlgorithmSpec{adaptFromTask<TRDDigitReaderSpec>(channels, useMC)},
                           Options{
                             {"digitsfile", VariantType::String, "trddigits.root", {"Input data file containing run3 digitizer going into Trap Simulator"}},
                             {"run2digitsfile", VariantType::String, "run2digits.root", {"Input data file containing run2 digitis going into Trap Simulator"}}}};
};

} //end namespace trd
} //end namespace o2
