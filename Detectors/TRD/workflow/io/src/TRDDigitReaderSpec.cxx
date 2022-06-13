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

using namespace o2::framework;

namespace o2
{
namespace trd
{

void TRDDigitReaderSpec::init(InitContext& ic)
{

  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("digitsfile"));
  connectTree();
}

void TRDDigitReaderSpec::connectTree()
{
  mTreeDigits.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(mFileName.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTreeDigits.reset((TTree*)mFile->Get(mDigitTreeName.c_str()));
  assert(mTreeDigits);
  mTreeDigits->SetBranchAddress(mDigitBranchName.c_str(), &mDigitsPtr);
  mTreeDigits->SetBranchAddress(mTriggerRecordBranchName.c_str(), &mTriggerRecordsPtr);
  if (mUseMC) {
    mTreeDigits->SetBranchAddress(mMCLabelsBranchName.c_str(), &mLabels);
  }
  LOG(info) << "Loaded tree from " << mFileName << " with " << mTreeDigits->GetEntries() << " entries";
}

void TRDDigitReaderSpec::run(ProcessingContext& pc)
{
  auto currEntry = mTreeDigits->GetReadEntry() + 1;
  assert(currEntry < mTreeDigits->GetEntries()); // this should not happen
  mTreeDigits->GetEntry(currEntry);
  LOG(info) << "Pushing " << mTriggerRecords.size() << " TRD trigger records at entry " << currEntry;
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRKTRGRD", 1, Lifetime::Timeframe}, mTriggerRecords);
  LOG(info) << "Pushing " << mDigits.size() << " digits for these trigger records";
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "DIGITS", 1, Lifetime::Timeframe}, mDigits);
  if (mUseMC) {
    auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{o2::header::gDataOriginTRD, "LABELS", 0, Lifetime::Timeframe});
    mLabels->copyandflatten(sharedlabels);
  }
  if (mTreeDigits->GetReadEntry() + 1 >= mTreeDigits->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getTRDDigitReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRD", "DIGITS", 1, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "TRKTRGRD", 1, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("TRD", "LABELS", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{"TRDDIGITREADER",
                           Inputs{},
                           outputs,
                           AlgorithmSpec{adaptFromTask<TRDDigitReaderSpec>(useMC)},
                           Options{
                             {"digitsfile", VariantType::String, "trddigits.root", {"Input data file containing TRD digits"}},
                             {"input-dir", VariantType::String, "none", {"Input directory"}}}};
};

} //end namespace trd
} //end namespace o2
