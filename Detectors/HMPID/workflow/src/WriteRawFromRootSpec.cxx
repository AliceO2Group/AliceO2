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

#include "Framework/DataRefUtils.h"
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
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCommonDataFormats/NameConf.h"

#include "TFile.h"
#include "TTree.h"
#include <TSystem.h>
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "HMPIDBase/Digit.h"
#include "HMPIDBase/Trigger.h"
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
  LOG(INFO) << "HMPID Write Raw File From Root sim Digits vector - init()";
  mDigitsReceived = 0;
  mEventsReceived = 0;
  mBaseRootFileName = ic.options().get<std::string>("in-file");
  mBaseFileName = ic.options().get<std::string>("outfile");
  mDirectoryName = ic.options().get<std::string>("outdir");
  mFileFor = ic.options().get<std::string>("file-for");
  mDumpDigits = ic.options().get<bool>("dump-digits"); // Debug flags
  mSkipEmpty = ic.options().get<bool>("skip-empty");

  if (gSystem->AccessPathName(mDirectoryName.c_str())) {
    if (gSystem->mkdir(mDirectoryName.c_str(), kTRUE)) {
      LOG(FATAL) << "could not create output directory " << mDirectoryName;
    } else {
      LOG(INFO) << "created output directory " << mDirectoryName;
    }
  }
  std::string fullFName = o2::utils::concat_string(mDirectoryName, "/", mBaseFileName);

  // Setup the Coder
  mCod = new HmpidCoder2(Geo::MAXEQUIPMENTS);
  mCod->setSkipEmptyEvents(mSkipEmpty);
  mCod->openOutputStream(fullFName, mFileFor);
  std::string inputGRP = o2::base::NameConf::getGRPFileName();
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom(inputGRP)};
  mCod->getWriter().setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::HMP)); // must be set explicitly

  // Open the ROOT file
  TFile* fdig = TFile::Open(mBaseRootFileName.data());
  assert(fdig != nullptr);
  LOG(INFO) << "Open Root digits file " << mBaseRootFileName.data();
  mDigTree = (TTree*)fdig->Get("o2sim");

  // Ready to operate
  mCod->getWriter().writeConfFile("HMP", "RAWDATA", o2::utils::concat_string(mDirectoryName, '/', "HMPraw.cfg"));
  mExTimer.start();
}

void WriteRawFromRootTask::readRootFile()
{
  std::vector<o2::hmpid::Digit> digitsPerEvent;
  std::vector<o2::hmpid::Digit> digits, *hmpBCDataPtr = &digits;
  std::vector<o2::hmpid::Trigger> interactions, *interactionsPtr = &interactions;

  // Keeps the Interactions !
  mDigTree->SetBranchAddress("InteractionRecords", &interactionsPtr);
  LOG(DEBUG) << "Number of Interaction Records vectors in the simulation file :" << mDigTree->GetEntries();
  for (int ient = 0; ient < mDigTree->GetEntries(); ient++) {
    mDigTree->GetEntry(ient);
    LOG(INFO) << "Interactions records in simulation :" << interactions.size();
    for (auto a : interactions) {
      LOG(DEBUG) << a;
    }
  }
  sort(interactions.begin(), interactions.end()); // Sort interactions in ascending order
  int trigPointer = 0;

  mDigTree->SetBranchAddress("HMPDigit", &hmpBCDataPtr);
  LOG(INFO) << "Number of entries in the simulation file :" << mDigTree->GetEntries();

  // Loops in the Entry of ROOT Branch
  for (int ient = 0; ient < mDigTree->GetEntries(); ient++) {
    mDigTree->GetEntry(ient);
    int nbc = digits.size();
    if (nbc == 0) { // exit for empty
      LOG(INFO) << "The Entry :" << ient << " doesn't have digits !";
      continue;
    }
    sort(digits.begin(), digits.end(), o2::hmpid::Digit::eventEquipPadsComp);
    if (mDumpDigits) { // we wand the dump of digits ?
      std::ofstream dumpfile;
      dumpfile.open("/tmp/hmpDumpDigits.dat");
      for (int i = 0; i < nbc; i++) {
        dumpfile << digits[i] << std::endl;
      }
      dumpfile.close();
    }
    // ready to operate
    LOG(INFO) << "For the entry = " << ient << " there are " << nbc << " DIGITS stored.";
    for (int i = 0; i < nbc; i++) {
      if (digits[i].getOrbit() != interactions[trigPointer].getOrbit() || digits[i].getBC() != interactions[trigPointer].getBc()) {
        do {
          mEventsReceived++;
          LOG(DEBUG) << "Orbit =" << interactions[trigPointer].getOrbit() << " BC =" << interactions[trigPointer].getBc();
          mCod->codeEventChunkDigits(digitsPerEvent, interactions[trigPointer]);
          digitsPerEvent.clear();
          trigPointer++;
        } while ((digits[i].getOrbit() != interactions[trigPointer].getOrbit() || digits[i].getBC() != interactions[trigPointer].getBc()) && trigPointer < interactions.size());
        if (trigPointer == interactions.size()) {
          LOG(WARNING) << "Digits without Interaction Record !!!  ABORT";
          break;
        }
      }
      digitsPerEvent.push_back(digits[i]);
    }
    mEventsReceived++;
    LOG(DEBUG) << "Orbit =" << interactions[trigPointer].getOrbit() << " BC =" << interactions[trigPointer].getBc();
    mCod->codeEventChunkDigits(digitsPerEvent, interactions[trigPointer]);
    mDigitsReceived += nbc;
  }
  mExTimer.logMes("End of Write raw file Job !");
  return;
}

void WriteRawFromRootTask::run(framework::ProcessingContext& pc)
{
  // Arrange Files path
  readRootFile();
  mCod->closeOutputStream();
  mCod->dumpResults(mBaseRootFileName);
  mExTimer.logMes("Raw File created ! Digits = " + std::to_string(mDigitsReceived) + " for Events =" + std::to_string(mEventsReceived));
  mExTimer.stop();
  pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
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
    Options{{"outdir", VariantType::String, "./", {"base dir for output file"}},
            {"file-for", VariantType::String, "all", {"single file per: all,flp,link"}},
            {"outfile", VariantType::String, "hmpid", {"base name for output file"}},
            {"in-file", VariantType::String, "hmpiddigits.root", {"name of the input sim root file"}},
            {"dump-digits", VariantType::Bool, false, {"out the digits file in /tmp/hmpDumpDigits.dat"}},
            {"skip-empty", VariantType::Bool, false, {"skip empty events"}}}};
}

} // namespace hmpid
} // end namespace o2
