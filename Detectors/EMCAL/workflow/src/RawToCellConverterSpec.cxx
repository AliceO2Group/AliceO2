// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <string>

#include "FairLogger.h"

#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsEMCAL/EMCALBlockHeader.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "DetectorsRaw/RDHUtils.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/Mapper.h"
#include "EMCALReconstruction/CaloFitResults.h"
#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloRawFitterStandard.h"
#include "EMCALReconstruction/CaloRawFitterGamma2.h"
#include "EMCALReconstruction/AltroDecoder.h"
#include "EMCALWorkflow/RawToCellConverterSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

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

  auto fitmethod = ctx.options().get<std::string>("fitmethod");
  if (fitmethod == "standard") {
    LOG(INFO) << "Using standard raw fitter";
    mRawFitter = std::unique_ptr<o2::emcal::CaloRawFitter>(new o2::emcal::CaloRawFitterStandard);
  } else if (fitmethod == "gamma2") {
    mRawFitter = std::unique_ptr<o2::emcal::CaloRawFitter>(new o2::emcal::CaloRawFitterGamma2);
  }

  mRawFitter->setAmpCut(mNoiseThreshold);
  mRawFitter->setL1Phase(0.);
}

void RawToCellConverterSpec::run(framework::ProcessingContext& ctx)
{
  LOG(DEBUG) << "[EMCALRawToCellConverter - run] called";
  const double CONVADCGEV = 0.016; // Conversion from ADC counts to energy: E = 16 MeV / ADC

  // Cache cells from for bunch crossings as the component reads timeframes from many links consecutively
  std::map<o2::InteractionRecord, std::shared_ptr<std::vector<o2::emcal::Cell>>> cellBuffer; // Internal cell buffer
  std::map<o2::InteractionRecord, uint32_t> triggerBuffer;

  int firstEntry = 0;
  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs())) {
    //o2::emcal::RawReaderMemory<o2::header::RAWDataHeaderV4> rawreader(gsl::span(rawData.payload, o2::framework::DataRefUtils::getPayloadSize(rawData)));

    o2::emcal::RawReaderMemory rawreader(o2::framework::DataRefUtils::as<const char>(rawData));

    // loop over all the DMA pages
    while (rawreader.hasNext()) {

      rawreader.next();

      auto& header = rawreader.getRawHeader();
      auto triggerBC = o2::raw::RDHUtils::getTriggerBC(header);
      auto triggerOrbit = o2::raw::RDHUtils::getTriggerOrbit(header);
      auto feeID = o2::raw::RDHUtils::getFEEID(header);
      auto triggerbits = o2::raw::RDHUtils::getTriggerType(header);

      o2::InteractionRecord currentIR(triggerBC, triggerOrbit);
      std::shared_ptr<std::vector<o2::emcal::Cell>> currentCellContainer;
      auto found = cellBuffer.find(currentIR);
      if (found == cellBuffer.end()) {
        currentCellContainer = std::make_shared<std::vector<o2::emcal::Cell>>();
        cellBuffer[currentIR] = currentCellContainer;
        // also add trigger bits
        triggerBuffer[currentIR] = triggerbits;
      } else {
        currentCellContainer = found->second;
      }

      if (feeID > 40) {
        continue; //skip STU ddl
      }

      //std::cout<<rawreader.getRawHeader()<<std::endl;

      // use the altro decoder to decode the raw data, and extract the RCU trailer
      o2::emcal::AltroDecoder decoder(rawreader);
      decoder.decode();

      LOG(DEBUG) << decoder.getRCUTrailer();

      o2::emcal::Mapper map = mMapper->getMappingForDDL(feeID);
      int iSM = feeID / 2;

      // Loop over all the channels
      for (auto& chan : decoder.getChannels()) {

        int iRow, iCol;
        ChannelType_t chantype;
        try {
          iRow = map.getRow(chan.getHardwareAddress());
          iCol = map.getColumn(chan.getHardwareAddress());
          chantype = map.getChannelType(chan.getHardwareAddress());
        } catch (Mapper::AddressNotFoundException& ex) {
          std::cerr << ex.what() << std::endl;
          continue;
        };

        int CellID = mGeometry->GetAbsCellIdFromCellIndexes(iSM, iRow, iCol);

        // define the conatiner for the fit results, and perform the raw fitting using the stadnard raw fitter
        o2::emcal::CaloFitResults fitResults = mRawFitter->evaluate(chan.getBunches(), 0, 0);
        if (fitResults.getAmp() < 0) {
          fitResults.setAmp(0.);
        }
        if (fitResults.getTime() < 0) {
          fitResults.setTime(0.);
        }
        currentCellContainer->emplace_back(CellID, fitResults.getAmp() * CONVADCGEV, fitResults.getTime(), chantype);
      }
    }
  }

  // Loop over BCs, sort cells with increasing tower ID and write to output containers
  mOutputCells.clear();
  mOutputTriggerRecords.clear();
  for (auto [bc, cells] : cellBuffer) {
    mOutputTriggerRecords.emplace_back(bc, triggerBuffer[bc], mOutputCells.size(), cells->size());
    if (cells->size()) {
      // Sort cells according to cell ID
      std::sort(cells->begin(), cells->end(), [](o2::emcal::Cell& lhs, o2::emcal::Cell& rhs) { return lhs.getTower() < rhs.getTower(); });
      for (auto cell : *cells) {
        mOutputCells.push_back(cell);
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

  outputs.emplace_back("EMC", "CELLS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("EMC", "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"EMCALRawToCellConverterSpec",
                                          o2::framework::select("A:EMC/RAWDATA"),
                                          outputs,
                                          o2::framework::adaptFromTask<o2::emcal::reco_workflow::RawToCellConverterSpec>(),
                                          o2::framework::Options{
                                            {"fitmethod", o2::framework::VariantType::String, "standard", {"Fit method (standard or gamma2)"}}}};
}
