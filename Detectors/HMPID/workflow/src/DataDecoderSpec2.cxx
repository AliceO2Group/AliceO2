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

/// \file   DataDecoderSpec.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
/// \brief Implementation of a data processor to run the HMPID raw decoding
///

#include <random>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>
#include <functional>
#include <vector>

#include "TTree.h"
#include "TFile.h"

#include <gsl/span>

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
#include "Framework/DataRefUtils.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDReconstruction/HmpidDecoder2.h"
#include "HMPIDWorkflow/DataDecoderSpec2.h"
#include "CommonUtils/VerbosityConfig.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

//=======================
// Data decoder
void DataDecoderTask2::init(framework::InitContext& ic)
{

  LOG(info) << "[HMPID Data Decoder - Init] ( create Raw Stream Decoder for " << Geo::MAXEQUIPMENTS << " equipments !";

  mProduceResults = ic.options().get<bool>("get-results-statistics");
  mRootStatFile = ic.options().get<std::string>("result-file");
  mFastAlgorithm = ic.options().get<bool>("fast-decode");
  mDeco = new o2::hmpid::HmpidDecoder2(Geo::MAXEQUIPMENTS);
  mDeco->init();
  mTotalDigits = 0;
  mTotalFrames = 0;

  mExTimer.start();
  return;
}

void DataDecoderTask2::run(framework::ProcessingContext& pc)
{
  mDeco->mDigits.clear();
  mTriggers.clear();
  LOG(info) << "[HMPID Data Decoder - Run] !";

  decodeTF(pc);
  //  TODO: accept other types of Raw Streams ...
  //  decodeReadout(pc);
  // decodeRawFile(pc);

  // Output the Digits/Triggers vector
  orderTriggers();
  pc.outputs().snapshot(o2::framework::Output{"HMP", "DIGITS", 0}, mDeco->mDigits);
  pc.outputs().snapshot(o2::framework::Output{"HMP", "INTRECORDS", 0}, mTriggers);

  mExTimer.elapseMes("Decoding... Digits decoded = " + std::to_string(mTotalDigits) + " Frames received = " + std::to_string(mTotalFrames));
  return;
}

void DataDecoderTask2::endOfStream(framework::EndOfStreamContext& ec)
{
  // Records the statistics
  float avgEventSize;    //[o2::hmpid::Geo::MAXEQUIPMENTS];
  float avgBusyTime;     //[o2::hmpid::Geo::MAXEQUIPMENTS];
  float numOfSamples;    //[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];
  float sumOfCharges;    //[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];
  float squareOfCharges; //[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];
  float xb;
  float yb;

  if (!mProduceResults) {
    LOG(info) << "Skip the Stat file creation ! ";
  } else {
    TString filename = TString::Format("%s_stat.root", mRootStatFile.c_str());
    LOG(info) << "Create the stat file " << filename.Data();
    TFile mfileOut(TString::Format("%s", filename.Data()), "RECREATE");
    TTree* theObj[Geo::N_MODULES + 1];
    for (int i = 0; i < Geo::N_MODULES; i++) { // Create the TTree array
      TString tit = TString::Format("HMPID Data Decoding Statistic results Mod=%d", i);
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

    // Update the Stat for the Decoding
    int numEqui = mDeco->getNumberOfEquipments();
    // cycle in order to update info for the last event
    for (int i = 0; i < numEqui; i++) {
      if (mDeco->mTheEquipments[i]->mNumberOfEvents > 0) {
        mDeco->updateStatistics(mDeco->mTheEquipments[i]);
      }
    }
    char summaryFileName[254];
    snprintf(summaryFileName, 254, "%s_stat.txt", mRootStatFile.c_str());
    mDeco->writeSummaryFile(summaryFileName);
    for (int e = 0; e < numEqui; e++) {
      avgEventSize = mDeco->getAverageEventSize(e);
      avgBusyTime = mDeco->getAverageBusyTime(e);
      theObj[Geo::N_MODULES]->Fill();
    }
    for (int m = 0; m < o2::hmpid::Geo::N_MODULES; m++) {
      for (int y = 0; y < o2::hmpid::Geo::N_YCOLS; y++) {
        for (int x = 0; x < o2::hmpid::Geo::N_XROWS; x++) {
          xb = x;
          yb = y;
          numOfSamples = mDeco->getPadSamples(m, x, y);
          sumOfCharges = mDeco->getPadSum(m, x, y);
          squareOfCharges = mDeco->getPadSquares(m, x, y);
          theObj[m]->Fill();
        }
      }
    }
    for (int i = 0; i <= Geo::N_MODULES; i++) {
      theObj[i]->Write();
    }
  }
  mExTimer.logMes("End the Decoding ! Digits decoded = " + std::to_string(mTotalDigits) + " Frames received = " + std::to_string(mTotalFrames));
  mExTimer.stop();
  return;
}
//_________________________________________________________________________________________________
// the decodeTF() function processes the the messages generated by the (sub)TimeFrame builder
void DataDecoderTask2::decodeTF(framework::ProcessingContext& pc)
{
  LOG(info) << "*********** In decodeTF **************";

  // get the input buffer
  auto& inputs = pc.inputs();

  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  {
    static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
    std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{"HMP", "RAWDATA", 0xDEADBEEF}}};
    for (const auto& ref : InputRecordWalker(inputs, dummy)) {
      const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadSize = DataRefUtils::getPayloadSize(ref);
      if (payloadSize == 0) {
        auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
        if (++contDeadBeef <= maxWarn) {
          LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
               dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
               contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
        }
        return;
      }
    }
    contDeadBeef = 0; // if good data, reset the counter
  }

  DPLRawParser parser(inputs, o2::framework::select("TF:HMP/RAWDATA"));
  // mDeco->mDigits.clear();
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    int pointerToTheFirst = mDeco->mDigits.size();
    uint32_t* theBuffer = (uint32_t*)it.raw();
    mDeco->setUpStream(theBuffer, it.size() + it.offset());
    try {
      if (mFastAlgorithm) {
        mDeco->decodePageFast(&theBuffer);
      } else {
        mDeco->decodePage(&theBuffer);
      }
    } catch (int e) {
      // The stream end !
      LOG(debug) << "End Page decoding !";
    }
    // std::cout << "  fDigit=" << pointerToTheFirst << " lDigit=," << mDeco->mDigits.size() << " nDigit=" << mDeco->mDigits.size()-pointerToTheFirst << std::endl;
    mTriggers.push_back(o2::hmpid::Trigger(mDeco->mIntReco, pointerToTheFirst, mDeco->mDigits.size() - pointerToTheFirst));
    mTotalFrames++;
  }

  mTotalDigits += mDeco->mDigits.size();
  LOG(info) << "Writing   Digitis=" << mDeco->mDigits.size() << "/" << mTotalDigits << " Frame=" << mTotalFrames << " IntRec " << mDeco->mIntReco;
  return;
}

//_________________________________________________________________________________________________
// the decodeReadout() function processes the messages generated by o2-mch-cru-page-reader-workflow
// TODO: rearrange, test
void DataDecoderTask2::decodeReadout(framework::ProcessingContext& pc)
{
  LOG(info) << "*********** In decode readout **************";

  // get the input buffer
  auto& inputs = pc.inputs();
  DPLRawParser parser(inputs, o2::framework::select("readout:HMP/RAWDATA"));
  //  DPLRawParser parser(inputs, o2::framework::select("HMP/readout"));

  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    uint32_t* theBuffer = (uint32_t*)it.raw();
    mDeco->setUpStream(theBuffer, it.size() + it.offset());
    try {
      if (mFastAlgorithm) {
        mDeco->decodePageFast(&theBuffer);
      } else {
        mDeco->decodePage(&theBuffer);
      }
    } catch (int e) {
      // The stream end !
      LOG(debug) << "End Page decoding !";
    }
  }
  return;
}

// the decodeReadout() function processes the messages generated by o2-mch-cru-page-reader-workflow
// TODO: rearrange, test
void DataDecoderTask2::decodeRawFile(framework::ProcessingContext& pc)
{
  LOG(info) << "*********** In decode rawfile **************";

  for (auto&& input : pc.inputs()) {
    if (input.spec->binding == "file") {

      auto const* raw = input.payload;
      size_t payloadSize = DataRefUtils::getPayloadSize(input);

      LOG(info) << "  payloadSize=" << payloadSize;
      if (payloadSize == 0) {
        return;
      }

      uint32_t* theBuffer = (uint32_t*)input.payload;
      int pagesize = payloadSize;
      mDeco->setUpStream(theBuffer, pagesize);
      try {
        if (mFastAlgorithm) {
          mDeco->decodePageFast(&theBuffer);
        } else {
          mDeco->decodePage(&theBuffer);
        }
      } catch (int e) {
        // The stream end !
        LOG(debug) << "End Page decoding !";
      }
    }
  }
  return;
}

void DataDecoderTask2::orderTriggers()
{
  std::vector<o2::hmpid::Digit> dig;
  dig.clear();
  std::vector<o2::hmpid::Trigger> trg;
  trg.clear();

  // first arrange the triggers in chronological order
  std::sort(mTriggers.begin(), mTriggers.end());
  // then build a new Digit Vector physically ordered for triggers
  int i = 0;
  int k = i;
  o2::hmpid::Trigger tr;
  int count = 0;
  int firstEntry;
  while (i < mTriggers.size()) {
    tr = mTriggers[i];
    count = 0;
    firstEntry = dig.size();
    while (k < mTriggers.size() && mTriggers[i].getTriggerID() == mTriggers[k].getTriggerID()) {
      for (int j = mTriggers[k].getFirstEntry(); j <= mTriggers[k].getLastEntry(); j++) {
        dig.push_back(mDeco->mDigits[j]);
        count++;
      }
      k++;
    }
    tr.setDataRange(firstEntry, count);
    trg.push_back(tr);
    i = k;
  }

  // then arrange the triggers in chamber order
  for (int i = 0; i < trg.size(); i++) {
    if (trg[i].getFirstEntry() > trg[i].getLastEntry()) {
      continue;
    }
    std::sort(dig.begin() + trg[i].getFirstEntry(), dig.begin() + trg[i].getLastEntry());
  }

  mTriggers.swap(trg);
  mDeco->mDigits.swap(dig);
  trg.clear();
  dig.clear();
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getDecodingSpec2(bool askDISTSTF)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("TF", o2::framework::ConcreteDataTypeMatcher{"HMP", "RAWDATA"}, o2::framework::Lifetime::Timeframe);
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("HMP", "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("HMP", "INTRECORDS", 0, o2::framework::Lifetime::Timeframe);

  return DataProcessorSpec{
    "HMP-RawStreamDecoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<DataDecoderTask2>()},
    Options{{"get-results-statistics", VariantType::Bool, false, {"Generate intermediat output results."}},
            {"result-file", VariantType::String, "/tmp/hmpRawDecodeResults", {"Base name of the decoding results files."}},
            {"fast-decode", VariantType::Bool, false, {"Use the fast algorithm. (error 0.8%)"}}}};
}

} // namespace hmpid
} // end namespace o2
