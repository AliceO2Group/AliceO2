// draft
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    DatDecoderSpec.cxx
/// \author  
///
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

#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DPLUtils/DPLRawParser.h"

#include "HMPIDBase/Digit.h"
#include "HMPIDBase/Geo.h"
#include "HMPIDReconstruction/HmpidDecodeRawMem.h"
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

  LOG(INFO) << "[HMPID Data Decoder - Init] ( create Decoder for " << Geo::MAXEQUIPMENTS << " equipments !";
  std::cout << "[HMPID Data Decoder - Init] ( create Decoder for " << Geo::MAXEQUIPMENTS << " equipments !" << std::endl;

  mDeco = new o2::hmpid::HmpidDecodeRawDigit(Geo::MAXEQUIPMENTS);
  mDeco->init();

  return;
}

void DataDecoderTask::run(framework::ProcessingContext& pc)
{
  mDeco->mDigits.clear();

  decodeTF(pc);
//  decodeReadout(pc);
//  decodeRawFile(pc);

  LOG(INFO) << "[HMPID Data Decoder - run] Writing " << mDeco->mDigits.size() << " Digits ...";
  pc.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginHMP, "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mDeco->mDigits);

  float avgEventSize[o2::hmpid::Geo::MAXEQUIPMENTS];
  float avgBusyTime[o2::hmpid::Geo::MAXEQUIPMENTS];

  uint32_t numOfSamples[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];
  double sumOfCharges[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];
  double squareOfCharges[o2::hmpid::Geo::N_MODULES][o2::hmpid::Geo::N_YCOLS][o2::hmpid::Geo::N_XROWS];

  TString filename = TString::Format("%s_%06d.root", "test", 1);
 // LOG(DEBUG) << "opening file " << filename.Data();
//--  std::unique_ptr<TFile> mfileOut = nullptr;
//--  mfileOut.reset(TFile::Open(TString::Format("%s", filename.Data()), "RECREATE"));

  //std::unique_ptr<TTree> theObj;
//  TTree *  theObj;
  //theObj = std::make_unique<TTree>("o2hmp", "HMPID Data Decoding Statistic results");
//  theObj = new TTree("o2hmp", "HMPID Data Decoding Statistic results");

//  theObj->Branch("Average_Event_Size", avgEventSize,"f[14]");
//  theObj->Branch("Average_Busy_Time", avgBusyTime,"f[14]");
//  theObj->Branch("Samples_per_pad", avgBusyTime,"d[7][144][160]");
//  theObj->Branch("Sum_of_charges_per_pad", sumOfCharges,"d[7][144][160]");
//  theObj->Branch("Sum_of_square_of_charges", squareOfCharges,"d[7][144][160]");

  int numEqui = mDeco->getNumberOfEquipments();
  for(int e=0;e<numEqui;e++) {
      avgEventSize[e] = mDeco->getAverageEventSize(e);
      avgBusyTime[e] = mDeco->getAverageBusyTime(e);
  }
  for(int m=0; m < o2::hmpid::Geo::N_MODULES; m++)
    for(int y=0; y < o2::hmpid::Geo::N_YCOLS; y++)
      for(int x=0; x < o2::hmpid::Geo::N_XROWS; x++ ) {
        numOfSamples[m][y][x] = mDeco->getPadSamples(m, x, y);
        sumOfCharges[m][y][x] = mDeco->getPadSum(m, x, y);
        squareOfCharges[m][y][x] = mDeco->getPadSquares(m, x, y);
  //      std::cout << "@ " << m <<","<<y<<","<<x << " "<< numOfSamples[m][y][x] << "=" <<sumOfCharges[m][y][x] << std::endl;
      }
//  theObj->Fill();

//-- mfileOut->WriteObject((TTree *)theObj, "HMPID Decoding Statistics");
//  mfileOut->cd();
//  theObj->Write();
//  theObj.reset();
//  mfileOut.reset();

 // pc.outputs().snapshot(o2::framework::Output{"HMP", "STATS", 0, o2::framework::Lifetime::Timeframe}, *theObj);

 //--- theObj->Reset();
 //---- mfileOut.reset();

}

/*    auto& digits = mDecoder->getOutputDigits();
  auto& orbits = mDecoder->getOrbits();

  if (mPrint) {
    for (auto d : digits) {
      std::cout << " DE# " << d.getDetID() << " PadId " << d.getPadID() << " ADC " << d.getADC() << " time " << d.getTime().sampaTime << std::endl;
    }
  }
  // send the output buffer via DPL
  size_t digitsSize, orbitsSize;
  char* digitsBuffer = createBuffer(digits, digitsSize);
  char* orbitsBuffer = createBuffer(orbits, orbitsSize);

  // create the output message
  auto freefct = [](void* data, void*) { free(data); };
//  pc.outputs().adoptChunk(Output{"MCH", "DIGITS", 0}, digitsBuffer, digitsSize, freefct, nullptr);
//  pc.outputs().adoptChunk(Output{"MCH", "ORBITS", 0}, orbitsBuffer, orbitsSize, freefct, nullptr);
*/

//_________________________________________________________________________________________________
// the decodeTF() function processes the the messages generated by the (sub)TimeFrame builder
void DataDecoderTask::decodeTF(framework::ProcessingContext& pc)
{
  LOG(INFO) << "*********** In decodeTF **************";

  // get the input buffer
  auto& inputs = pc.inputs();
  DPLRawParser parser(inputs, o2::framework::select("TF:HMP/RAWDATA"));

//  auto& digitsV = pc.outputs().make<std::vector<o2::hmpid::Digit>>(Output{"HMP", "DIGITS", 0, Lifetime::Timeframe});

  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
 //   mDeco->mDigits.clear();

    uint32_t *theBuffer = (uint32_t *)it.raw();
  //  std::cout << "Decode parser loop :"<< it.size() << " , " << it.offset() << std::endl;
    mDeco->setUpStream(theBuffer, it.size()+it.offset());
    mDeco->decodePageFast(&theBuffer);
//    for(auto d : mDeco->mDigits)
//      digitsV.push_back(d);

 //   pc.outputs().snapshot(o2::framework::Output{"HMP", "DIGITS", 0, o2::framework::Lifetime::Timeframe}, mDeco->mDigits);

  }
  return;
}
//pc.outputs().make
//_________________________________________________________________________________________________
// the decodeReadout() function processes the messages generated by o2-mch-cru-page-reader-workflow
void DataDecoderTask::decodeReadout(framework::ProcessingContext& pc)
{
  LOG(INFO) << "*********** In decode readout **************";

  // get the input buffer
  auto& inputs = pc.inputs();
  DPLRawParser parser(inputs, o2::framework::select("readout:HMP/RAWDATA"));
//  DPLRawParser parser(inputs, o2::framework::select("HMP/readout"));

  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    uint32_t *theBuffer = (uint32_t *)it.raw();
    mDeco->setUpStream(theBuffer, it.size()+it.offset());
    mDeco->decodePageFast(&theBuffer);
  }
  return;
}

// the decodeReadout() function processes the messages generated by o2-mch-cru-page-reader-workflow
void DataDecoderTask::decodeRawFile(framework::ProcessingContext& pc)
{
  LOG(INFO) << "*********** In decode rawfile **************";

  for (auto&& input : pc.inputs()) {
    if (input.spec->binding == "rawfile") {
  //    const auto* header = o2::header::get<header::DataHeader*>(input.header);
  //    if (!header) {
  //      return;
  //    }
      const o2::header::DataHeader* header = o2::header::get<header::DataHeader*>(input.header);
      uint32_t *theBuffer = (uint32_t *)input.payload;
      int pagesize = header->payloadSize;
      std::cout << "Get page !" << std::endl;
      mDeco->setUpStream(theBuffer, pagesize);
      mDeco->decodePageFast(&theBuffer);
    }
  }
  return;
}

//_________________________________________________________________________________________________
o2::framework::DataProcessorSpec getDecodingSpec(std::string inputSpec)
//o2::framework::DataPrecessorSpec getDecodingSpec()
{
  
  std::vector<o2::framework::InputSpec> inputs;
//  inputs.emplace_back("TF", o2::header::gDataOriginHMP, "RAWDATA", 0, Lifetime::Timeframe);

  inputs.emplace_back("TF", o2::framework::ConcreteDataTypeMatcher{"HMP", "RAWDATA"}, o2::framework::Lifetime::Timeframe);

 // inputs.emplace_back("rawfile", o2::framework::ConcreteDataTypeMatcher{"HMP", "RAWDATA"}, o2::framework::Lifetime::Timeframe);
//  inputs.emplace_back("readout", o2::header::gDataOriginHMP, "RAWDATA", 0, Lifetime::Timeframe);
//  inputs.emplace_back("readout", o2::header::gDataOriginHMP, "RAWDATA", 0, Lifetime::Timeframe);
//  inputs.emplace_back("rawfile", o2::header::gDataOriginHMP, "RAWDATA", 0, Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("HMP", "DIGITS", 0, o2::framework::Lifetime::Timeframe);
 // outputs.emplace_back("HMP", "ORBITS", 0, o2::framework::Lifetime::Timeframe);
 // outputs.emplace_back("HMP", "STATS", 0, o2::framework::Lifetime::Timeframe);

  
  return DataProcessorSpec{
    "HMP-DataDecoder",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<DataDecoderTask>()},
    Options{{"print", VariantType::Bool, false, {"print digits"}}} };
}

} // namespace hmpid
} // end namespace o2
