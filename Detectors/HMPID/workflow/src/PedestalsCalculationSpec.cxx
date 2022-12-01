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
#include "TString.h"
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
#include "Framework/RawDeviceService.h"
#include <fairmq/Device.h>

#include "CCDB/CcdbApi.h"
#include "CCDB/CCDBTimeStampUtils.h"

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"
#include "CommonUtils/NameConf.h"

#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDReconstruction/HmpidDecoder2.h"
#include "HMPIDWorkflow/PedestalsCalculationSpec.h"

#include "DataFormatsDCS/DCSConfigObject.h"
namespace o2
{
namespace hmpid
{

using namespace o2;
using namespace o2::framework;
using RDH = o2::header::RDHAny;

//=======================
// Init
void PedestalsCalculationTask::init(framework::InitContext& ic)
{

  LOG(info) << "[HMPID Pedestal Calculation - v.1 - Init] ( create Decoder for " << Geo::MAXEQUIPMENTS << " equipments !";

  mDeco = new o2::hmpid::HmpidDecoder2(Geo::MAXEQUIPMENTS);
  mDeco->init();
  mTotalDigits = 0;
  mTotalFrames = 0;

  mSigmaCut = ic.options().get<float>("sigmacut");
  mWriteToFiles = ic.options().get<bool>("use-files");
  mPedestalsBasePath = ic.options().get<std::string>("files-basepath");
  mPedestalsCCDBBasePath = mPedestalsBasePath;

  mWriteToDB = ic.options().get<bool>("use-ccdb");
  if (mWriteToDB) {
    mDBapi.init(ic.options().get<std::string>("ccdb-uri")); // or http://localhost:8080 for a local installation
    mWriteToDB = mDBapi.isHostReachable() ? true : false;
  }

  mPedestalTag = ic.options().get<std::string>("pedestals-tag");
  mFastAlgorithm = ic.options().get<bool>("fast-decode");

  mWriteToDCSDB = ic.options().get<bool>("use-dcsccdb");
  if (mWriteToDCSDB) {
    mDCSDBapi.init(ic.options().get<std::string>("dcsccdb-uri")); // or http://localhost:8080 for a local installation
    mWriteToDCSDB = mDCSDBapi.isHostReachable() ? true : false;
  }
  mDcsCcdbAliveHours = ic.options().get<int>("dcsccdb-alivehours");

  mExTimer.start();
  LOG(info) << "Calculate Ped/Thresh." + (mWriteToDB ? " Store in DCSCCDB at " + mPedestalsBasePath + " with Tag:" + mPedestalTag : " CCDB not used !");
  return;
}

void PedestalsCalculationTask::run(framework::ProcessingContext& pc)
{
  if (mPedestalTag == "run_number") { // if the Tag is run_number, then substitute the Tag with RN
    const std::string NAStr = "NA";
    mPedestalTag = pc.services().get<RawDeviceService>().device()->fConfig->GetProperty<std::string>("runNumber", NAStr);
  }
  decodeTF(pc);
  mExTimer.elapseMes("Decoding... Digits decoded = " + std::to_string(mTotalDigits) + " Frames received = " + std::to_string(mTotalFrames));
  return;
}

void PedestalsCalculationTask::endOfStream(framework::EndOfStreamContext& ec)
{
  if (mWriteToDB) {
    recordPedInCcdb();
  }
  if (mWriteToDCSDB) {
    recordPedInDcsCcdb();
  }
  if (mWriteToFiles) {
    recordPedInFiles();
  }
  mExTimer.stop();
  return;
}

void PedestalsCalculationTask::recordPedInFiles()
{
  double Average;
  double Variance;
  double Samples;
  double SumOfCharge;
  double SumOfSquares;

  uint32_t Buffer;
  uint32_t Pedestal;
  uint32_t Threshold;

  for (int e = 0; e < Geo::MAXEQUIPMENTS; e++) {
    if (mDeco->getAverageEventSize(e) == 0) {
      continue;
    }
    auto padsFileName = fmt::format("{}_{}.dat", mPedestalsBasePath, std::to_string(e));
    FILE* fpads = fopen(padsFileName.c_str(), "w");
    if (fpads == nullptr) {
      mExTimer.logMes("error creating the file = " + std::string(padsFileName));
      LOG(error) << "error creating the file = " << padsFileName;
      return;
    }
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
  return;
}

void PedestalsCalculationTask::recordPedInDcsCcdb()
{
  // create the root structure
  LOG(info) << "Store Pedestals in DCS CCDB ";

  float xb, yb, ch, Samples;
  double SumOfCharge, SumOfSquares, Average, Variance;
  uint32_t Pedestal, Threshold, PedThr;
  std::string PedestalFixedTag = "Latest";

  o2::dcs::DCSconfigObject_t pedestalsConfig;

// Setup dimensions for Equipment granularity with 48 channels/dilogic
#define PEDTHFORMAT "%05X,"
#define COLUMNTAIL false
#define PEDTHBYTES 6
  int bufferDim = PEDTHBYTES * Geo::N_CHANNELS * Geo::N_DILOGICS * Geo::N_COLUMNS + 10;
  char* outBuffer = new char[bufferDim];
  char* inserPtr;
  char* endPtr = outBuffer + bufferDim;

  for (int e = 0; e < Geo::MAXEQUIPMENTS; e++) {
    if (mDeco->getAverageEventSize(e) == 0) { // skip the empty equipment
      continue;
    }
    inserPtr = outBuffer;
    // algoritm based on equipment granularity
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
          PedThr = ((Threshold & 0x001FF) << 9) | (Pedestal & 0x001FF);
          assert(inserPtr < endPtr);
          snprintf(inserPtr, endPtr - inserPtr, PEDTHFORMAT, PedThr);
          inserPtr += PEDTHBYTES;
        }
        if (COLUMNTAIL) {
          for (int h = 48; h < 64; h++) {
            assert(inserPtr < endPtr);
            snprintf(inserPtr, endPtr - inserPtr, PEDTHFORMAT, 0);
            inserPtr += PEDTHBYTES;
          }
        }
      }
    }
    mExTimer.logMes("End write the equipment = " + std::to_string(e));
    assert(inserPtr < endPtr);
    snprintf(inserPtr, endPtr - inserPtr, "%05X\n", 0xA0A0A); // The closure value
    inserPtr += 6;
    *inserPtr = '\0'; // close the string rap.
    o2::dcs::addConfigItem(pedestalsConfig, "Equipment" + std::to_string(e), (const char*)outBuffer);
  }

  long minTimeStamp = o2::ccdb::getCurrentTimestamp();
  long maxTimeStamp = minTimeStamp + (3600L * mDcsCcdbAliveHours * 1000);

  auto filename = fmt::format("{}_{}.dat", mPedestalsBasePath, PedestalFixedTag);
  mExTimer.logMes("File name = >" + filename + "< (" + mPedestalsCCDBBasePath + "," + PedestalFixedTag);

  mDbMetadata.emplace("Tag", PedestalFixedTag.c_str());
  mDCSDBapi.storeAsTFileAny(&pedestalsConfig, filename.c_str(), mDbMetadata, minTimeStamp, maxTimeStamp);

  mExTimer.logMes("End Writing the pedestals ! Digits decoded = " + std::to_string(mTotalDigits) + " Frames received = " + std::to_string(mTotalFrames));

  return;
}

void PedestalsCalculationTask::recordPedInCcdb()
{
  // create the root structure
  LOG(info) << "Store Pedestals in ccdb ";

  float xb, yb, ch;
  double Samples, SumOfCharge, SumOfSquares, Average, Variance;

  TObjArray aSigmas(Geo::N_MODULES);
  TObjArray aPedestals(Geo::N_MODULES);

  for (int i = 0; i < Geo::N_MODULES; i++) {
    aSigmas.AddAt(new TMatrixF(Geo::N_XROWS, Geo::N_YCOLS), i);
    aPedestals.AddAt(new TMatrixF(Geo::N_XROWS, Geo::N_YCOLS), i);
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

  long minTimeStamp = o2::ccdb::getCurrentTimestamp();
  long maxTimeStamp = minTimeStamp + (3600L * 24 * (5 * 365) * 1000); // 5 years

  for (int i = 0; i < Geo::N_MODULES; i++) {
    if (mDeco->getAverageEventSize(i * 2) == 0 && mDeco->getAverageEventSize(i * 2 + 1) == 0) {
      continue; // If no events skip the chamber
    }
    TString filename = TString::Format("%s/%s/Mean_%d", mPedestalsCCDBBasePath.c_str(), mPedestalTag.c_str(), i);
    mDbMetadata.emplace("Tag", mPedestalTag.c_str());
    mDBapi.storeAsTFileAny(aPedestals.At(i), filename.Data(), mDbMetadata, minTimeStamp, maxTimeStamp);
  }
  for (int i = 0; i < Geo::N_MODULES; i++) {
    if (mDeco->getAverageEventSize(i * 2) == 0 && mDeco->getAverageEventSize(i * 2 + 1) == 0) {
      continue; // If no events skip the chamber
    }
    TString filename = TString::Format("%s/%s/Sigma_%d", mPedestalsCCDBBasePath.c_str(), mPedestalTag.c_str(), i);
    mDbMetadata.emplace("Tag", mPedestalTag.c_str());
    mDBapi.storeAsTFileAny(aSigmas.At(i), filename.Data(), mDbMetadata, minTimeStamp, maxTimeStamp);
  }
  return;
}

//_________________________________________________________________________________________________
// the decodeTF() function processes the the messages generated by the (sub)TimeFrame builder
void PedestalsCalculationTask::decodeTF(framework::ProcessingContext& pc)
{
  LOG(debug) << "*********** In decodeTF **************";
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
      LOG(debug) << "End Page decoding !";
    }
    mTotalFrames++;
    mTotalDigits += mDeco->mDigits.size();
  }
  return;
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getPedestalsCalculationSpec(std::string inputSpec)
{

  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("TF", o2::framework::ConcreteDataTypeMatcher{"HMP", "RAWDATA"}, o2::framework::Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;

  return DataProcessorSpec{
    "HMP-PestalsCalculation",
    o2::framework::select(inputSpec.c_str()),
    outputs,
    AlgorithmSpec{adaptFromTask<PedestalsCalculationTask>()},
    Options{{"files-basepath", VariantType::String, "HMP/Config", {"Name of the Base Path of Pedestals/Thresholds files."}},
            {"use-files", VariantType::Bool, false, {"Register the Pedestals/Threshold values into ASCII files"}},
            {"use-ccdb", VariantType::Bool, false, {"Register the Pedestals/Threshold values into the CCDB"}},
            {"ccdb-uri", VariantType::String, "http://ccdb-test.cern.ch:8080", {"URI for the CCDB access."}},
            {"use-dcsccdb", VariantType::Bool, false, {"Register the Pedestals/Threshold values into the DCS-CCDB"}},
            {"dcsccdb-uri", VariantType::String, "http://ccdb-test.cern.ch:8080", {"URI for the DCS-CCDB access."}},
            {"dcsccdb-alivehours", VariantType::Int, 3, {"Alive hours in DCS-CCDB."}},
            {"fast-decode", VariantType::Bool, true, {"Use the fast algorithm. (error 0.8%)"}},
            {"pedestals-tag", VariantType::String, "Latest", {"The tag applied to this set of pedestals/threshold values"}},
            {"sigmacut", VariantType::Float, 4.0f, {"Sigma values for the Thresholds calculation."}}}};
}

} // namespace hmpid
} // end namespace o2
