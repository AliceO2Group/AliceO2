// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "FairLogger.h"

#include "DataFormatsEMCAL/EMCALBlockHeader.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "EMCALWorkflow/RawToCellConverterSpec.h"
#include "Framework/ControlService.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/Mapper.h"
#include "EMCALReconstruction/CaloFitResults.h"
#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloRawFitterStandard.h"
#include "EMCALReconstruction/AltroDecoder.h"
#include "CommonDataFormat/InteractionRecord.h"

using namespace o2::emcal::reco_workflow;

void RawToCellConverterSpec::init(framework::InitContext& ctx)
{
  LOG(DEBUG) << "[EMCALRawToCellConverter - init] Initialize converter ";
  if (!mGeometry) {
    mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(223409);
  }
  if (!mGeometry) {
    LOG(ERROR) << "Failure accessing geometry";
  }

  if (!mMapper) {
    mMapper = std::unique_ptr<o2::emcal::MappingHandler>(new o2::emcal::MappingHandler);
  }
  if (!mMapper) {
    LOG(ERROR) << "Failed to initialize mapper";
  }

  mRawFitter.setAmpCut(mNoiseThreshold);
  mRawFitter.setL1Phase(0.);
}

void RawToCellConverterSpec::run(framework::ProcessingContext& ctx)
{
  LOG(DEBUG) << "[EMCALRawToCellConverter - run] called";

  mOutputCells.clear();
  mOutputTriggerRecords.clear();

  int firstEntry = 0;
  for (const auto& rawData : ctx.inputs()) {

    //o2::emcal::RawReaderMemory<o2::header::RAWDataHeaderV4> rawreader(gsl::span(rawData.payload, o2::framework::DataRefUtils::getPayloadSize(rawData)));

    o2::emcal::RawReaderMemory<o2::header::RAWDataHeaderV4> rawreader(o2::framework::DataRefUtils::as<const char>(rawData));

    bool first = true;
    uint16_t currentTrigger = 0;
    uint32_t currentorbit = 0;

    // loop over all the DMA pages
    while (rawreader.hasNext()) {

      rawreader.next();

      auto header = rawreader.getRawHeader();

      if (!first) { // check if it is the first event in the payload
        std::cout << " triggerBC " << header.triggerBC << " current Trigger " << currentTrigger << std::endl;
        if (header.triggerBC > currentTrigger) { //new event
          mOutputTriggerRecords.emplace_back(o2::InteractionRecord(currentTrigger, currentorbit), firstEntry, mOutputCells.size() - 1);
          firstEntry = mOutputCells.size();

          currentTrigger = header.triggerBC;
          currentorbit = header.triggerOrbit;
        }      //new event
      } else { //first
        currentTrigger = header.triggerBC;
        std::cout << " first is true and I set triggerBC to currentTrigger " << currentTrigger << std::endl;
        currentorbit = header.triggerOrbit;
        std::cout << " and set first to false " << std::endl;
        first = false;
      }

      if (header.feeId > 40)
        continue; //skip STU ddl

      //std::cout<<rawreader.getRawHeader()<<std::endl;

      // use the altro decoder to decode the raw data, and extract the RCU trailer
      o2::emcal::AltroDecoder<decltype(rawreader)> decoder(rawreader);
      decoder.decode();

      std::cout << decoder.getRCUTrailer() << std::endl;

      o2::emcal::Mapper map = mMapper->getMappingForDDL(header.feeId);

      // Loop over all the channels
      for (auto& chan : decoder.getChannels()) {

        int iRow = map.getRow(chan.getHardwareAddress());
        int iCol = map.getColumn(chan.getHardwareAddress());
        ChannelType_t chantype = map.getChannelType(chan.getHardwareAddress());
        int iSM = header.feeId / 2;

        int CellID = mGeometry->GetAbsCellIdFromCellIndexes(iSM, iRow, iCol);

        // define the conatiner for the fit results, and perform the raw fitting using the stadnard raw fitter
        o2::emcal::CaloFitResults fitResults = mRawFitter.evaluate(chan.getBunches(), 0, 0);

        if (fitResults.getAmp() < 0 && fitResults.getTime() < 0) {
          fitResults.setAmp(0.);
          fitResults.setTime(0.);
        }
        mOutputCells.emplace_back(CellID, fitResults.getAmp(), fitResults.getTime(), chantype);
      }
    }
  }

  LOG(DEBUG) << "[EMCALRawToCellConverter - run] Writing " << mOutputCells.size() << " cells ...";
  ctx.outputs().snapshot(o2::framework::Output{"EMC", "CELLS", 0, o2::framework::Lifetime::Timeframe}, mOutputCells);
  ctx.outputs().snapshot(o2::framework::Output{"EMC", "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe}, mOutputTriggerRecords);
}

o2::framework::DataProcessorSpec o2::emcal::reco_workflow::getRawToCellConverterSpec()
{
  std::vector<o2::framework::InputSpec> inputs;
  std::vector<o2::framework::OutputSpec> outputs;

  inputs.emplace_back("readout-proxy", "FLP", "RAWDATA", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("EMC", "CELLS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("EMC", "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"EMCALRawToCellConverterSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::emcal::reco_workflow::RawToCellConverterSpec>()};
}
