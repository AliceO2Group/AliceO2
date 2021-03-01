// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   WriteRawFromDigitsSpec.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
/// \brief Implementation of a data processor to produce raw files from a Digits stream
///

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>
#include <functional>
#include <vector>
#include <algorithm>

#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Logger.h"
#include "Framework/InputRecordWalker.h"

#include "TFile.h"
#include "TTree.h"


#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "HMPIDBase/Digit.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDSimulation/HmpidCoder2.h"
#include "HMPIDWorkflow/WriteRawFromRootSpec.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

//=======================
// Data decoder
void WriteRawFromRootTask::init(framework::InitContext& ic)
{
  LOG(INFO) << "[HMPID Write Raw File From Root sim Digits vector - init()]";
  mBaseFileName = ic.options().get<std::string>("out-file");
  mBaseRootFileName = ic.options().get<std::string>("in-file");
  mSkipEmpty = ic.options().get<bool>("skip-empty");
  mDigitsReceived = 0;
  mEventsReceived = 0;
  mDumpDigits = ic.options().get<bool>("dump-digits");
  mPerFlpFile = ic.options().get<bool>("per-flp-file");

  mCod = new HmpidCoder2(Geo::MAXEQUIPMENTS);
//  mCod->reset();
  mCod->openOutputStream(mBaseFileName.c_str(), mPerFlpFile);

  TFile* fdig = TFile::Open(mBaseRootFileName.data());
  assert(fdig != nullptr);
  LOG(INFO) << " Open Root digits file " << mBaseRootFileName.data();
  mDigTree = (TTree*)fdig->Get("o2sim");

  mExTimer.start();
  return;
}


void WriteRawFromRootTask::readRootFile()
{
  std::vector<o2::hmpid::Digit> digitsPerEvent;
  std::vector<o2::hmpid::Digit> digits,*hmpBCDataPtr = &digits;
//  std::vector<o2::ft0::ChannelData> digitsCh, *ft0ChDataPtr = &digitsCh;

  mDigTree->SetBranchAddress("HMPDigit", &hmpBCDataPtr);
  //  digTree->SetBranchAddress("FT0DIGITSCH", &ft0ChDataPtr);

  uint32_t old_orbit = ~0;
  uint32_t old_bc = ~0;

  LOG(INFO) << "Number of entries in the simulation file :" << mDigTree->GetEntries();
  for (int ient = 0; ient < mDigTree->GetEntries(); ient++) {
    mDigTree->GetEntry(ient);
    int nbc = digits.size();
    if (nbc == 0)
      continue; // exit for empty

    sort(digits.begin(), digits.end(), o2::hmpid::Digit::eventEquipPadsComp);
    if (mDumpDigits) {
      std::ofstream dumpfile;
      dumpfile.open("/tmp/hmpDumpDigits.dat");
      for (int i = 0; i < nbc; i++) {
        dumpfile << digits[i] << std::endl;
      }
      dumpfile.close();
    }

    //    sort(digits.begin(), digits.end(), o2::hmpid::Digit::eventEquipPadsComp);
    LOG(INFO) << "For the entry = " << ient << " there are " << nbc << " BCs stored.";

    old_orbit = digits[0].getOrbit();
    old_bc = digits[0].getBC();
    mEventsReceived++;
    LOG(INFO) << "Orbit = " << old_orbit << " BC " << old_bc;
    for (int i = 0; i < nbc; i++) {
      if (digits[i].getOrbit() != old_orbit || digits[i].getBC() != old_bc) { // the event is finished
        mCod->codeEventChunkDigits(digitsPerEvent);
        digitsPerEvent.clear();
        old_orbit = digits[i].getOrbit();
        old_bc = digits[i].getBC();
        mEventsReceived++;
        LOG(INFO) << "Orbit = " << old_orbit << " BC " << old_bc;
      }
      digitsPerEvent.push_back(digits[i]);
    }
    if (digitsPerEvent.size() > 0) {
      mCod->codeEventChunkDigits(digitsPerEvent);
    }
    mDigitsReceived += nbc;
  }
}

void WriteRawFromRootTask::run(framework::ProcessingContext& pc)
{
  readRootFile();

  mExTimer.logMes("End Of Job !");
  mCod->closeOutputStream();
  mCod->dumpResults();
  mExTimer.logMes("Raw File created ! Digits = " + std::to_string(mDigitsReceived) + " for Events =" + std::to_string(mEventsReceived));
  mExTimer.stop();

  pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
  return;
}

void WriteRawFromRootTask::endOfStream(framework::EndOfStreamContext& ec)
{
  return;
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getWriteRawFromRootSpec(std::string inputSpec)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;

  return DataProcessorSpec{
    "HMP-WriteRawFromRootFile",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<WriteRawFromRootTask>()},
    Options{{"out-file", VariantType::String, "hmpidRaw", {"name of the output file"}},
            {"in-file", VariantType::String, "simulation.root", {"name of the input sim root file"}},
            {"per-flp-file", VariantType::Bool, false, {"produce one raw file per FLPs"}},
            {"dump-digits", VariantType::Bool, false, {"out the digits file in /tmp/hmpDumpDigits.dat"}},
            {"skip-empty", VariantType::Bool, false, {"skip empty events"}}}};
}

} // namespace hmpid
} // end namespace o2
