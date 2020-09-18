// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDWorkflow/TRDDigitReaderSpec.h"

#include <cstdlib>
// this is somewhat assuming that a DPL workflow will run on one node

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "DPLUtils/RootTreeReader.h"
#include "Headers/DataHeader.h"
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include <SimulationDataFormat/IOMCTruthContainerView.h>
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"
#include "TRDBase/Digit.h" // for the Digit type
#include "TRDSimulation/TrapSimulator.h"
#include "TRDSimulation/Digitizer.h"
#include "TRDSimulation/Detector.h" // for the Hit type

#include "DetectorsBase/GeometryManager.h"

#include "DataFormatsTRD/TriggerRecord.h"

#include <TTree.h>
#include <TFile.h>
#include <TSystem.h>
#include <TRandom1.h>

#include <sstream>
#include <cmath>
#include <unistd.h>   // for getppid
#include <TMessage.h> // object serialization
#include <memory>     // std::unique_ptr
#include <cstring>    // memcpy
#include <string>     // std::string
#include <cassert>
#include <chrono>
#include <thread>
#include <algorithm>

using namespace o2::framework;
using SubSpecificationType = o2::framework::DataAllocator::SubSpecificationType;

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
    getFromBranch(mMCLabelsBranchName.c_str(), (void**)&ioLabels);

    // publish labels in shared memory
    auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::trd::MCLabel>>(Output{"TRD", "LABELS", 0, Lifetime::Timeframe});
    ioLabels->copyandflatten(sharedlabels);
    pc.outputs().snapshot(Output{"TRD", "DIGITS", 0, Lifetime::Timeframe}, *digits);
    pc.outputs().snapshot(Output{"TRD", "TRGRDIG", 0, Lifetime::Timeframe}, *triggerRecords);
    LOG(info) << "TRDDigitReader digits size=" << digits->size() << " triggerrecords size=" << triggerRecords->size() << " mc labels size (in bytes) = " << sharedlabels.size();
  }
  //delete DPLTree; // next line will delete the pointer as well.
  mFile->Close();

  mState = 2; // prevent coming in here again.
              // send endOfData control event and mark the reader as ready to finish
  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getTRDDigitReaderSpec(int channels)
{

  return DataProcessorSpec{"TRDDIGITREADER",
                           Inputs{},
                           Outputs{
                             OutputSpec{"TRD", "DIGITS", 0, Lifetime::Timeframe},
                             OutputSpec{"TRD", "TRGRDIG", 0, Lifetime::Timeframe},
                             OutputSpec{"TRD", "LABELS", 0, Lifetime::Timeframe}},
                           // outputs,
                           AlgorithmSpec{adaptFromTask<TRDDigitReaderSpec>(channels)},
                           Options{
                             {"digitsfile", VariantType::String, "trddigits.root", {"Input data file containing run3 digitizer going into Trap Simulator"}},
                             {"run2digitsfile", VariantType::String, "run2digits.root", {"Input data file containing run2 digitis going into Trap Simulator"}}}};
};

} //end namespace trd
} //end namespace o2
