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

#include "TRDWorkflowIO/TRDConfigEventReaderSpec.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "CommonUtils/StringUtils.h"
#include "fairlogger/Logger.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

void TRDConfigEventReaderSpec::init(InitContext& ic)
{

  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("configeventsfile"));
  connectTree();
}

void TRDConfigEventReaderSpec::connectTree()
{
  mTreeConfigEvent.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(mFileName.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTreeConfigEvent.reset((TTree*)mFile->Get(mConfigEventTreeName.c_str()));
  assert(mTreeConfigEvent);
  mTreeConfigEvent->SetBranchAddress(mConfigEventBranchName.c_str(), &mTrapConfigEventPtr);
  LOG(info) << "Loaded tree from " << mFileName << " with a trapconfig";
}

void TRDConfigEventReaderSpec::run(ProcessingContext& pc)
{
  auto currEntry = mTreeConfigEvent->GetReadEntry() + 1;
  assert(currEntry < mTreeConfigEvent->GetEntries()); // this should not happen
  mTreeConfigEvent->GetEntry(currEntry);
  LOG(info) << "Pushing Trapconfig to file ";
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "CONFEVT", 1}, mTrapConfigEvent);
  //  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "CONFEVT", 1, Lifetime::Timeframe}, mTrapConfigEvent);
  if (mTreeConfigEvent->GetReadEntry() + 1 >= mTreeConfigEvent->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getTRDConfigEventReaderSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRD", "CFGEVT", 1, Lifetime::Timeframe);
  return DataProcessorSpec{"TRDCONFIGEVENTREADER",
                           Inputs{},
                           outputs,
                           AlgorithmSpec{adaptFromTask<TRDConfigEventReaderSpec>()},
                           Options{
                             {"configeventfile", VariantType::String, "trdconfigevent.root", {"Input Configuration event"}},
                             {"input-dir", VariantType::String, "none", {"Input directory"}}}};
};

} // end namespace trd
} // end namespace o2
