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

/// \file   RawToDigitsSpec.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 18 mar 2021
/// \brief Implementation of a data processor to produce Digits from Raw files
///

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>
#include <functional>
#include <vector>
#include <algorithm>

#include "DPLUtils/DPLRawParser.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"

#include "TTree.h"
#include "TFile.h"

#include <gsl/span>

#include "Framework/InputSpec.h"
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

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RawFileReader.h"
#include "DetectorsRaw/RDHUtils.h"

#include "CommonDataFormat/InteractionRecord.h"

#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDReconstruction/HmpidDecoder2.h"
#include "HMPIDWorkflow/RawToDigitsSpec.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::framework;
using RDH = o2::header::RDHAny;
using namespace o2::hmpid::raw;

//=======================
// Data decoder
void RawToDigitsTask::init(framework::InitContext& ic)
{
  LOG(info) << "[HMPID Write Root File From Raw Files - init()]";

  // get line command options
  mOutRootFileName = ic.options().get<std::string>("out-file");
  mBaseFileName = ic.options().get<std::string>("base-file");
  mInputRawFileName = ic.options().get<std::string>("in-file");
  mFastAlgorithm = ic.options().get<bool>("fast-decode");
  mDigitsReceived = 0;
  mFramesReceived = 0;

  mReader.setDefaultDataOrigin(o2::header::gDataOriginHMP);
  mReader.setDefaultDataDescription(o2::header::gDataDescriptionRawData);
  mReader.setDefaultReadoutCardType(o2::raw::RawFileReader::RORC);
  mReader.addFile(mInputRawFileName);
  mReader.init();

  mDecod = new o2::hmpid::HmpidDecoder2(Geo::MAXEQUIPMENTS);
  mDecod->init();

  mExTimer.start();
  return;
}

void RawToDigitsTask::run(framework::ProcessingContext& pc)
{
  bool isInLoop = true;
  int tfID;
  std::vector<char> dataBuffer; // where to put extracted data

  if (mReader.getNTimeFrames() == 0) {
    parseNoTF();
    isInLoop = false;
  }

  while (isInLoop) {
    tfID = mReader.getNextTFToRead();
    if (tfID >= mReader.getNTimeFrames()) {
      LOG(info) << "nothing left to read after " << tfID << " TFs read";
      break;
    }
    for (int il = 0; il < mReader.getNLinks(); il++) {
      auto& link = mReader.getLink(il);
      LOG(info) << "Decoding link " << il;
      auto sz = link.getNextTFSize(); // size in char needed for the next TF of this link
      dataBuffer.resize(sz);
      link.readNextTF(dataBuffer.data());
      link.rewindToTF(tfID);
      int nhbf = link.getNHBFinTF();
      LOG(debug) << " Number of HBF " << nhbf;
      for (int ib = 0; ib < nhbf; ib++) {
        auto zs = link.getNextHBFSize(); // size in char needed for the next TF of this link
        dataBuffer.resize(zs);
        link.readNextHBF(dataBuffer.data());
        // Parse
        uint32_t* ptrBuffer = (uint32_t*)dataBuffer.data();
        uint32_t* ptrBufferEnd = ptrBuffer + zs / 4;
        mDecod->setUpStream(ptrBuffer, zs);
        while (ptrBuffer < ptrBufferEnd) {
          try {
            if (mFastAlgorithm) {
              mDecod->decodePageFast(&ptrBuffer);
            } else {
              mDecod->decodePage(&ptrBuffer);
            }
          } catch (int e) {
            // The stream end !
            LOG(debug) << "End Page decoding !";
          }
          int first = mAccumulateDigits.size();
          mAccumulateDigits.insert(mAccumulateDigits.end(), mDecod->mDigits.begin(), mDecod->mDigits.end());
          int last = mAccumulateDigits.size();
          if (last > first) {
            mEvents.emplace_back(o2::hmpid::Trigger{mDecod->mIntReco, first, last - first});
            mDigitsReceived += mDecod->mDigits.size();
          }
          mFramesReceived++;
          LOG(debug) << "run() Digits received =" << mDigitsReceived << " frames = " << mFramesReceived << " size=" << sz << " F-L " << first << "," << last << " " << mDecod->mIntReco;
          mDecod->mDigits.clear();
        }
      }
    }
    mReader.setNextTFToRead(++tfID);
  }
  mTotalDigits += mDigitsReceived;
  mTotalFrames += mFramesReceived;

  mExTimer.logMes("End of Decoding ! Digits = " + std::to_string(mTotalDigits) + " Frames received = " + std::to_string(mTotalFrames));

  writeResults();

  //  pc.services().get<ControlService>().endOfStream();
  pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
  mExTimer.stop();
  return;
}

void RawToDigitsTask::endOfStream(framework::EndOfStreamContext& ec)
{
  mExTimer.logMes("Received an End Of Stream !");
  return;
}

void RawToDigitsTask::parseNoTF()
{
  std::vector<char> dataBuffer; // where to put extracted data

  for (int il = 0; il < mReader.getNLinks(); il++) {
    auto& link = mReader.getLink(il);
    LOG(info) << "Decoding link " << il;
    auto sz = link.getNextTFSize(); // size in char needed for the next TF of this link
    LOG(info) << " Size TF " << sz;
    dataBuffer.resize(sz);
    link.readNextTF(dataBuffer.data());

    uint32_t* ptrBuffer = (uint32_t*)dataBuffer.data();
    uint32_t* ptrBufferEnd = ptrBuffer + sz / 4;
    mDecod->setUpStream(ptrBuffer, sz);
    while (ptrBuffer < ptrBufferEnd) {
      try {
        if (mFastAlgorithm) {
          mDecod->decodePageFast(&ptrBuffer);
        } else {
          mDecod->decodePage(&ptrBuffer);
        }
      } catch (int e) {
        // The stream end !
        LOG(debug) << "End Fast Page decoding !";
      }
      int first = mAccumulateDigits.size();
      mAccumulateDigits.insert(mAccumulateDigits.end(), mDecod->mDigits.begin(), mDecod->mDigits.end());
      int last = mAccumulateDigits.size();
      if (last > first) {
        mEvents.emplace_back(mDecod->mIntReco, first, last - first);
        mDigitsReceived += mDecod->mDigits.size();
      }
      mFramesReceived++;
      LOG(info) << "run() Digits received =" << mDigitsReceived << " frames = " << mFramesReceived << " size=" << sz << " F-L " << first << "," << last << " " << mDecod->mIntReco;
      mDecod->mDigits.clear();
    }
  }
  return;
}

void RawToDigitsTask::writeResults()
{
  int numEqui = mDecod->getNumberOfEquipments(); // Update the Stat for the Decoding
  for (int i = 0; i < numEqui; i++) {
    if (mDecod->mTheEquipments[i]->mNumberOfEvents > 0) {
      mDecod->updateStatistics(mDecod->mTheEquipments[i]);
    }
  }
  if (mEvents.size() == 0) { // check if no evwnts
    LOG(info) << "There are not Event recorded ! Abort. ";
    mExTimer.stop();
    return;
  }
  for (int i = mEvents.size() - 1; i >= 0; i--) { // remove events that are (0,0) trigger
    if (mEvents[i].getTriggerID() == 0) {
      mEvents.erase(mEvents.begin() + i);
    }
  }
  sort(mEvents.begin(), mEvents.end()); // sort the events
  mExTimer.logMes("Sorted  Events = " + std::to_string(mEvents.size()));

  //  ---------- ROOT file version 2  ---------------
  TString filename;
  TString tit;

  std::vector<o2::hmpid::Digit> digitVec;
  std::vector<o2::hmpid::Trigger> eventVec;

  filename = TString::Format("%s", mOutRootFileName.c_str());
  LOG(info) << "Create the ROOT file " << filename.Data();
  TFile mfileOut(TString::Format("%s", filename.Data()), "RECREATE");
  tit = TString::Format("HMPID Raw File Decoding");
  TTree* theTree = new TTree("o2hmp", tit);

  theTree->Branch("InteractionRecords", &eventVec);
  theTree->Branch("HMPIDDigits", &digitVec);

  // builds the two arranged vectors of objects
  o2::hmpid::Trigger prevEvent = mEvents[0];
  uint32_t theFirstDigit = 0;
  uint32_t theLastDigit = 0;
  for (int e = 0; e < mEvents.size(); e++) {
    LOG(debug) << "Manage event " << mEvents[e];
    if (prevEvent != mEvents[e]) { // changes the event Flush It
      eventVec.emplace_back(o2::InteractionRecord(prevEvent.getBc(), prevEvent.getOrbit()), theFirstDigit, theLastDigit - theFirstDigit);
      theFirstDigit = theLastDigit;
      prevEvent = mEvents[e];
    }
    int first = mEvents[e].getFirstEntry();
    int last = mEvents[e].getLastEntry();
    for (int idx = first; idx <= last; idx++) {
      digitVec.push_back(mAccumulateDigits[idx]);
      theLastDigit++;
    }
  }
  eventVec.emplace_back(o2::InteractionRecord(prevEvent.getBc(), prevEvent.getOrbit()), theFirstDigit, theLastDigit - theFirstDigit);
  theTree->Fill();
  theTree->Write();
  mfileOut.Close();
  mExTimer.logMes("Register Tree ! ");

  //  ---------- Records the statistics -----------------
  float avgEventSize;    //[o2::hmpid::Geo::MAXEQUIPMENTS];
  float avgBusyTime;     //[o2::hmpid::Geo::MAXEQUIPMENTS];
  float numOfSamples;    //[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];
  float sumOfCharges;    //[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];
  float squareOfCharges; //[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];
  float xb;
  float yb;

  filename = TString::Format("%s_stat.root", mBaseFileName.c_str());
  LOG(info) << "Create the ROOT file " << filename.Data();
  TFile mfileOutStat(TString::Format("%s", filename.Data()), "RECREATE");
  TTree* theObj[Geo::N_MODULES + 1];
  for (int i = 0; i < Geo::N_MODULES; i++) { // Create the TTree array
    tit = TString::Format("HMPID Data Decoding Statistic results Mod=%d", i);
    theObj[i] = new TTree("o2hmp", tit);
    theObj[i]->Branch("x", &xb, "s");
    theObj[i]->Branch("y", &yb, "s");
    theObj[i]->Branch("Samples", &numOfSamples, "i");
    theObj[i]->Branch("Sum_of_charges", &sumOfCharges, "l");
    theObj[i]->Branch("Sum_of_square", &squareOfCharges, "l");
  }
  theObj[Geo::N_MODULES] = new TTree("o2hmp", "HMPID Data Decoding Statistic results");
  theObj[Geo::N_MODULES]->Branch("Average_Event_Size", &avgEventSize, "F");
  theObj[Geo::N_MODULES]->Branch("Average_Busy_Time", &avgBusyTime, "F");

  char summaryFileName[254];
  snprintf(summaryFileName, 254, "%s_stat.txt", mBaseFileName.c_str());
  mDecod->writeSummaryFile(summaryFileName);
  for (int e = 0; e < numEqui; e++) {
    avgEventSize = mDecod->getAverageEventSize(e);
    avgBusyTime = mDecod->getAverageBusyTime(e);
    theObj[Geo::N_MODULES]->Fill();
  }
  for (int m = 0; m < o2::hmpid::Geo::N_MODULES; m++) {
    for (int y = 0; y < o2::hmpid::Geo::N_YCOLS; y++) {
      for (int x = 0; x < o2::hmpid::Geo::N_XROWS; x++) {
        xb = x;
        yb = y;
        numOfSamples = mDecod->getPadSamples(m, x, y);
        sumOfCharges = mDecod->getPadSum(m, x, y);
        squareOfCharges = mDecod->getPadSquares(m, x, y);
        theObj[m]->Fill();
      }
    }
  }
  for (int i = 0; i <= Geo::N_MODULES; i++) {
    theObj[i]->Write();
  }

  mExTimer.logMes("End storing results ! Digits = " + std::to_string(mTotalDigits) + " Frames received = " + std::to_string(mTotalFrames));
  return;
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getRawToDigitsSpec(std::string inputSpec)
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;

  return DataProcessorSpec{
    "HMP-RawToDigits",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<RawToDigitsTask>()},
    Options{{"in-file", VariantType::String, "hmpidRaw.raw", {"name of the input Raw file"}},
            {"fast-decode", VariantType::Bool, false, {"Use the fast algorithm. (error 0.8%)"}},
            {"out-file", VariantType::String, "hmpReco.root", {"name of the output file"}},
            {"base-file", VariantType::String, "hmpDecode", {"base name for statistical output file"}}}};
}

} // namespace hmpid
} // end namespace o2
