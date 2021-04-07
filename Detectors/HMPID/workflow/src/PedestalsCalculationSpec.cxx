// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   PedestalsCalculationSpec.cxx
/// \author Antonio Franco - INFN Bari
/// \version 1.0
/// \date 01 feb 2021
/// \brief Implementation of a data processor to read a raw file and produce Pedestals/Threshold files
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
#include "TMath.h"
#include "TMatrixF.h"

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

#include "CCDB/CcdbApi.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDReconstruction/HmpidDecoder2.h"
#include "HMPIDWorkflow/PedestalsCalculationSpec.h"

namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

//=======================
// Data decoder
void PedestalsCalculationTask::init(framework::InitContext& ic)
{

  LOG(INFO) << "[HMPID Pedestal Calculation - Init] ( create Decoder for " << Geo::MAXEQUIPMENTS << " equipments !";

  mDeco = new o2::hmpid::HmpidDecoder2(Geo::MAXEQUIPMENTS);
  mDeco->init();
  mTotalDigits = 0;
  mTotalFrames = 0;

  mSigmaCut = ic.options().get<float>("sigmacut");
  mPedestalsBasePath = ic.options().get<std::string>("files-basepath");
  mPedestalTag = ic.options().get<std::string>("pedestals-tag");
  mDBapi.init(ic.options().get<std::string>("ccdb-uri")); // or http://localhost:8080 for a local installation
  mWriteToDB = mDBapi.isHostReachable() ? true : false;
  mFastAlgorithm = ic.options().get<bool>("fast-decode");

  mExTimer.start();
  return;
}

void PedestalsCalculationTask::run(framework::ProcessingContext& pc)
{
  decodeTF(pc);
  //  TODO: accept other types of Raw Streams ...
  //  decodeReadout(pc);
  // decodeRawFile(pc);ccdb

  mExTimer.elapseMes("Decoding... Digits decoded = " + std::to_string(mTotalDigits) + " Frames received = " + std::to_string(mTotalFrames));
  return;
}

void PedestalsCalculationTask::endOfStream(framework::EndOfStreamContext& ec)
{
  double Average;
  double Variance;
  double Samples;
  double SumOfCharge;
  double SumOfSquares;

  uint32_t Buffer;
  uint32_t Pedestal;
  uint32_t Threshold;
  char padsFileName[1024];

  for (int e = 0; e < Geo::MAXEQUIPMENTS; e++) {
    if (mDeco->getAverageEventSize(e) == 0) {
      continue;
    }
    sprintf(padsFileName, "%s_%d.dat", mPedestalsBasePath.c_str(), e);
    FILE* fpads = fopen(padsFileName, "w");
    // TODO: Add controls on the file open
    for (int c = 0; c < Geo::N_COLUMNS; c++) {
      for (int d = 0; d < Geo::N_DILOGICS; d++) {
        for (int h = 0; h < Geo::N_CHANNELS; h++) {
          Samples = (double)mDeco->getChannelSamples(e, c, d, h);
          SumOfCharge = mDeco->getChannelSum(e, c, d, h);
          SumOfSquares = mDeco->getChannelSquare(e, c, d, h);

          if (Samples > 0) {
            Average = SumOfCharge / Samples;
            Variance = sqrt(abs((Samples * SumOfSquares) - (SumOfCharge * SumOfCharge))) / Samples;
          } else {
            Average = 0;
            Variance = 0;
          }
          Pedestal = (uint32_t)Average;
          Threshold = (uint32_t)(Variance * mSigmaCut + Average);
          Buffer = ((Threshold & 0x001FF) << 9) | (Pedestal & 0x001FF);
          fprintf(fpads, "%05X\n", Buffer);
        }
        for (int h = 48; h < 64; h++) {
          fprintf(fpads, "%05X\n", 0);
        }
      }
    }
    mExTimer.logMes("End write the equipment = " + std::to_string(e));
    fprintf(fpads, "%05X\n", 0xA0A0A);
    fclose(fpads);
  }
  mExTimer.logMes("End Writing the pedestals ! Digits decoded = " + std::to_string(mTotalDigits) + " Frames received = " + std::to_string(mTotalFrames));

  if (mWriteToDB) {
    recordPedInCcdb();
  }
  mExTimer.stop();
  return;
}

void PedestalsCalculationTask::recordPedInCcdb()
{
  // create the root structure

  LOG(INFO) << "Store Pedestals in ccdb ";

  float xb, yb, ch;
  double Samples, SumOfCharge, SumOfSquares, Average, Variance;

  TObjArray aSigmas(Geo::N_MODULES);
  TObjArray aPedestals(Geo::N_MODULES);

  for (int i = 0; i < Geo::N_MODULES; i++) {
    aSigmas.AddAt(new TMatrixF(160, 144), i);
    aPedestals.AddAt(new TMatrixF(160, 144), i);
  }

  for (int m = 0; m < o2::hmpid::Geo::N_MODULES; m++) {
    if (mDeco->getAverageEventSize(m * 2) == 0 && mDeco->getAverageEventSize(m * 2 + 1) == 0) {
      continue; // If no events skip the chamber
    }
    TMatrixF* pS = (TMatrixF*)aSigmas.At(m);
    TMatrixF* pP = (TMatrixF*)aPedestals.At(m);

    for (int x = 0; x < o2::hmpid::Geo::N_XROWS; x++) {
      for (int y = 0; y < o2::hmpid::Geo::N_YCOLS; y++) {

        Samples = (double)mDeco->getPadSamples(m, x, y);
        SumOfCharge = mDeco->getPadSum(m, x, y);
        SumOfSquares = mDeco->getPadSquares(m, x, y);
        if (Samples > 0) {
          (*pP)(x, y) = SumOfCharge / Samples;
          (*pS)(x, y) = sqrt(abs((Samples * SumOfSquares) - (SumOfCharge * SumOfCharge))) / Samples;
        } else {
          (*pP)(x, y) = 0;
          (*pS)(x, y) = 0;
        }
      }
    }
  }
  struct timeval tp;
  gettimeofday(&tp, nullptr);
  uint64_t ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;

  uint64_t minTimeStamp = ms;
  uint64_t maxTimeStamp = ms + 1;

  for (int i = 0; i < Geo::N_MODULES; i++) {
    if (mDeco->getAverageEventSize(i * 2) == 0 && mDeco->getAverageEventSize(i * 2 + 1) == 0) {
      continue; // If no events skip the chamber
    }
    TString filename = TString::Format("HMP/Pedestals/%s/Mean_%d", mPedestalTag.c_str(), i);
    mDbMetadata.emplace("Tag", mPedestalTag.c_str());
    mDBapi.storeAsTFileAny(aPedestals.At(i), filename.Data(), mDbMetadata, minTimeStamp, maxTimeStamp);
  }
  for (int i = 0; i < Geo::N_MODULES; i++) {
    if (mDeco->getAverageEventSize(i * 2) == 0 && mDeco->getAverageEventSize(i * 2 + 1) == 0) {
      continue; // If no events skip the chamber
    }
    TString filename = TString::Format("HMP/Pedestals/%s/Sigma_%d", mPedestalTag.c_str(), i);
    mDbMetadata.emplace("Tag", mPedestalTag.c_str());
    mDBapi.storeAsTFileAny(aSigmas.At(i), filename.Data(), mDbMetadata, minTimeStamp, maxTimeStamp);
  }
  return;
}

//_________________________________________________________________________________________________
// the decodeTF() function processes the the messages generated by the (sub)TimeFrame builder
void PedestalsCalculationTask::decodeTF(framework::ProcessingContext& pc)
{
  LOG(DEBUG) << "*********** In decodeTF **************";

  // get the input buffer
  auto& inputs = pc.inputs();
  DPLRawParser parser(inputs, o2::framework::select("TF:HMP/RAWDATA"));
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
      LOG(DEBUG) << "End Fast Page decoding !";
    }
    mTotalFrames++;
    mTotalDigits += mDeco->mDigits.size();
  }
  return;
}
//pc.outputs().make
//_________________________________________________________________________________________________
// the decodeReadout() function processes the messages generated by o2-mch-cru-page-reader-workflow
void PedestalsCalculationTask::decodeReadout(framework::ProcessingContext& pc)
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
      LOG(DEBUG) << "End Fast Page decoding !";
    }
  }
  return;
}

// the decodeReadout() function processes the messages generated by o2-mch-cru-page-reader-workflow
void PedestalsCalculationTask::decodeRawFile(framework::ProcessingContext& pc)
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
        LOG(DEBUG) << "End Fast Page decoding !";
      }
    }
  }
  return;
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getPedestalsCalculationSpec(std::string inputSpec)
{

  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("TF", o2::framework::ConcreteDataTypeMatcher{"HMP", "RAWDATA"}, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("file", o2::framework::ConcreteDataTypeMatcher{"ROUT", "RAWDATA"}, o2::framework::Lifetime::Timeframe);
  //  inputs.emplace_back("readout", o2::header::gDataOriginHMP, "RAWDATA", 0, Lifetime::Timeframe);
  //  inputs.emplace_back("readout", o2::header::gDataOriginHMP, "RAWDATA", 0, Lifetime::Timeframe);
  //  inputs.emplace_back("rawfile", o2::header::gDataOriginHMP, "RAWDATA", 0, Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;
  //  outputs.emplace_back("HMP", "DIGITS", 0, o2::framework::Lifetime::Timeframe);

  return DataProcessorSpec{
    "HMP-DataDecoder",
    o2::framework::select(inputSpec.c_str()),
    outputs,
    AlgorithmSpec{adaptFromTask<PedestalsCalculationTask>()},
    Options{{"files-basepath", VariantType::String, "/tmp/hmpPedThr", {"Name of the Base Path of Pedestals/Thresholds files."}},
            {"use-ccdb", VariantType::Bool, false, {"Register the Pedestals/Threshold values into the CCDB"}},
            {"ccdb-uri", VariantType::String, "http://ccdb-test.cern.ch:8080", {"URI for the CCDB access."}},
            {"fast-decode", VariantType::Bool, false, {"Use the fast algorithm. (error 0.8%)"}},
            {"pedestals-tag", VariantType::String, "Latest", {"The tag applied to this set of pedestals/threshold values"}},
            {"sigmacut", VariantType::Float, 4.0f, {"Sigma values for the Thresholds calculation."}}}};
}

} // namespace hmpid
} // end namespace o2
