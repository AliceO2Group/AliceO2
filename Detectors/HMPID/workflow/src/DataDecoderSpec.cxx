// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDReconstruction/HmpidDecoder2.h"
#include "HMPIDWorkflow/DataDecoderSpec.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

//=======================
// Data decoder
void DataDecoderTask::init(framework::InitContext& ic)
{

  LOG(INFO) << "[HMPID Data Decoder - Init] ( create Raw Stream Decoder for " << Geo::MAXEQUIPMENTS << " equipments !";

  mRootStatFile = ic.options().get<std::string>("result-file");
  mFastAlgorithm = ic.options().get<bool>("fast-decode");
  mDeco = new o2::hmpid::HmpidDecoder2(Geo::MAXEQUIPMENTS);
  mDeco->init();
  mTotalDigits = 0;
  mTotalFrames = 0;

  mExTimer.start();
  return;
}

void DataDecoderTask::run(framework::ProcessingContext& pc)
{
  mDeco->mDigits.clear();
  decodeTF(pc);
  //  TODO: accept other types of Raw Streams ...
  //  decodeReadout(pc);
  // decodeRawFile(pc);

  pc.outputs().snapshot(o2::framework::Output{"HMP", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mDeco->mDigits);
  pc.outputs().snapshot(o2::framework::Output{"HMP", "INTRECORDS", 0, o2::framework::Lifetime::Timeframe}, mDeco->mIntReco);

  LOG(DEBUG) << "Writing   Digitis=" << mDeco->mDigits.size() << "/" << mTotalDigits << " Frame=" << mTotalFrames << " IntRec " << mDeco->mIntReco;
  mExTimer.elapseMes("Decoding... Digits decoded = " + std::to_string(mTotalDigits) + " Frames received = " + std::to_string(mTotalFrames));
}

void DataDecoderTask::endOfStream(framework::EndOfStreamContext& ec)
{
  // Records the statistics
  float avgEventSize;    //[o2::hmpid::Geo::MAXEQUIPMENTS];
  float avgBusyTime;     //[o2::hmpid::Geo::MAXEQUIPMENTS];
  float numOfSamples;    //[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];
  float sumOfCharges;    //[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];
  float squareOfCharges; //[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];
  float xb;
  float yb;

  TString filename = TString::Format("%s_stat.root", mRootStatFile.c_str());
  LOG(INFO) << "Create the stat file " << filename.Data();
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
  sprintf(summaryFileName, "%s_stat.txt", mRootStatFile.c_str());
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

  mExTimer.logMes("End the Decoding ! Digits decoded = " + std::to_string(mTotalDigits) + " Frames received = " + std::to_string(mTotalFrames));
  mExTimer.stop();
  return;
}
//_________________________________________________________________________________________________
// the decodeTF() function processes the the messages generated by the (sub)TimeFrame builder
void DataDecoderTask::decodeTF(framework::ProcessingContext& pc)
{
  LOG(DEBUG) << "*********** In decodeTF **************";

  // get the input buffer
  auto& inputs = pc.inputs();

  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  {
    std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{"HMP", "RAWDATA", 0xDEADBEEF}}};
    for (const auto& ref : InputRecordWalker(inputs, dummy)) {
      const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      if (dh->payloadSize == 0) {
        LOGP(WARNING, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize);
        return;
      }
    }
  }

  DPLRawParser parser(inputs, o2::framework::select("TF:HMP/RAWDATA"));
  mDeco->mDigits.clear();
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
      LOG(DEBUG) << "End Page decoding !";
    }
    mTotalFrames++;
  }
  mTotalDigits += mDeco->mDigits.size();
}

//_________________________________________________________________________________________________
// the decodeReadout() function processes the messages generated by o2-mch-cru-page-reader-workflow
// TODO: rearrange, test
void DataDecoderTask::decodeReadout(framework::ProcessingContext& pc)
{
  LOG(INFO) << "*********** In decode readout **************";

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
      LOG(DEBUG) << "End Page decoding !";
    }
  }
  return;
}

// the decodeReadout() function processes the messages generated by o2-mch-cru-page-reader-workflow
// TODO: rearrange, test
void DataDecoderTask::decodeRawFile(framework::ProcessingContext& pc)
{
  LOG(INFO) << "*********** In decode rawfile **************";

  for (auto&& input : pc.inputs()) {
    if (input.spec->binding == "file") {
      const header::DataHeader* header = o2::header::get<header::DataHeader*>(input.header);
      if (!header) {
        return;
      }

      auto const* raw = input.payload;
      size_t payloadSize = header->payloadSize;

      LOG(INFO) << "  payloadSize=" << payloadSize;
      if (payloadSize == 0) {
        return;
      }

      uint32_t* theBuffer = (uint32_t*)input.payload;
      int pagesize = header->payloadSize;
      mDeco->setUpStream(theBuffer, pagesize);
      try {
        if (mFastAlgorithm) {
          mDeco->decodePageFast(&theBuffer);
        } else {
          mDeco->decodePage(&theBuffer);
        }
      } catch (int e) {
        // The stream end !
        LOG(DEBUG) << "End Page decoding !";
      }
    }
  }
  return;
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getDecodingSpec(bool askDISTSTF)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("TF", o2::framework::ConcreteDataTypeMatcher{"HMP", "RAWDATA"}, o2::framework::Lifetime::Optional);
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
    AlgorithmSpec{adaptFromTask<DataDecoderTask>()},
    Options{{"result-file", VariantType::String, "/tmp/hmpRawDecodeResults", {"Base name of the decoding results files."}},
            {"fast-decode", VariantType::Bool, false, {"Use the fast algorithm. (error 0.8%)"}}}};
}

} // namespace hmpid
} // end namespace o2
