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
#include <string>

#include "FairLogger.h"

#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsEMCAL/EMCALBlockHeader.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "DataFormatsEMCAL/ErrorTypeFEE.h"
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

RawToCellConverterSpec::~RawToCellConverterSpec()
{
  if (mErrorMessagesSuppressed) {
    LOG(WARNING) << "Suppressed further " << mErrorMessagesSuppressed << " error messages";
  }
}

void RawToCellConverterSpec::init(framework::InitContext& ctx)
{
  LOG(DEBUG) << "[EMCALRawToCellConverter - init] Initialize converter ";
  if (!mGeometry) {
    mGeometry = Geometry::GetInstanceFromRunNumber(223409);
  }
  if (!mGeometry) {
    LOG(ERROR) << "Failure accessing geometry";
  }

  if (!mMapper) {
    mMapper = std::unique_ptr<MappingHandler>(new o2::emcal::MappingHandler);
  }
  if (!mMapper) {
    LOG(ERROR) << "Failed to initialize mapper";
  }

  auto fitmethod = ctx.options().get<std::string>("fitmethod");
  if (fitmethod == "standard") {
    LOG(INFO) << "Using standard raw fitter";
    mRawFitter = std::unique_ptr<CaloRawFitter>(new o2::emcal::CaloRawFitterStandard);
  } else if (fitmethod == "gamma2") {
    mRawFitter = std::unique_ptr<CaloRawFitter>(new o2::emcal::CaloRawFitterGamma2);
  }

  mMaxErrorMessages = ctx.options().get<int>("maxmessage");
  LOG(INFO) << "Suppressing error messages after " << mMaxErrorMessages << " messages";

  mRawFitter->setAmpCut(mNoiseThreshold);
  mRawFitter->setL1Phase(0.);
}

void RawToCellConverterSpec::run(framework::ProcessingContext& ctx)
{
  LOG(DEBUG) << "[EMCALRawToCellConverter - run] called";
  const double CONVADCGEV = 0.016; // Conversion from ADC counts to energy: E = 16 MeV / ADC
  constexpr auto originEMC = o2::header::gDataOriginEMC;
  constexpr auto descRaw = o2::header::gDataDescriptionRawData;

  mOutputCells.clear();
  mOutputTriggerRecords.clear();
  mOutputDecoderErrors.clear();

  if (isLostTimeframe(ctx)) {
    sendData(ctx, mOutputCells, mOutputTriggerRecords, mOutputDecoderErrors);
    return;
  }

  // Cache cells from for bunch crossings as the component reads timeframes from many links consecutively
  std::map<o2::InteractionRecord, std::shared_ptr<std::vector<Cell>>> cellBuffer; // Internal cell buffer
  std::map<o2::InteractionRecord, uint32_t> triggerBuffer;

  std::vector<framework::InputSpec> filter{{"filter", framework::ConcreteDataTypeMatcher(originEMC, descRaw)}};
  int firstEntry = 0;
  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs(), filter)) {

    // Skip SOX headers
    auto rdhblock = reinterpret_cast<const o2::header::RDHAny*>(rawData.payload);
    if (o2::raw::RDHUtils::getHeaderSize(rdhblock) == static_cast<int>(o2::framework::DataRefUtils::getPayloadSize(rawData))) {
      continue;
    }

    //o2::emcal::RawReaderMemory<o2::header::RAWDataHeaderV4> rawreader(gsl::span(rawData.payload, o2::framework::DataRefUtils::getPayloadSize(rawData)));

    o2::emcal::RawReaderMemory rawreader(framework::DataRefUtils::as<const char>(rawData));

    // loop over all the DMA pages
    while (rawreader.hasNext()) {

      rawreader.next();

      auto& header = rawreader.getRawHeader();
      auto triggerBC = raw::RDHUtils::getTriggerBC(header);
      auto triggerOrbit = raw::RDHUtils::getTriggerOrbit(header);
      auto feeID = raw::RDHUtils::getFEEID(header);
      auto triggerbits = raw::RDHUtils::getTriggerType(header);

      o2::InteractionRecord currentIR(triggerBC, triggerOrbit);
      std::shared_ptr<std::vector<Cell>> currentCellContainer;
      auto found = cellBuffer.find(currentIR);
      if (found == cellBuffer.end()) {
        currentCellContainer = std::make_shared<std::vector<Cell>>();
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
      AltroDecoder decoder(rawreader);
      //check the words of the payload exception in altrodecoder
      try {
        decoder.decode();
      } catch (AltroDecoderError& e) {
        std::string errormessage;
        using AltroErrType = AltroDecoderError::ErrorType_t;
        /// @TODO still need to add the RawFitter errors
        ErrorTypeFEE errornum(feeID, AltroDecoderError::errorTypeToInt(e.getErrorType()), -1);
        switch (e.getErrorType()) {
          case AltroErrType::RCU_TRAILER_ERROR:
            errormessage = " RCU Trailer Error ";
            break;
          case AltroErrType::RCU_VERSION_ERROR:
            errormessage = " RCU Version Error ";
            break;
          case AltroErrType::RCU_TRAILER_SIZE_ERROR:
            errormessage = " RCU Trailer Size Error ";
            break;
          case AltroErrType::ALTRO_BUNCH_HEADER_ERROR:
            errormessage = " ALTRO Bunch Header Error ";
            break;
          case AltroErrType::ALTRO_BUNCH_LENGTH_ERROR:
            errormessage = " ALTRO Bunch Length Error ";
            break;
          case AltroErrType::ALTRO_PAYLOAD_ERROR:
            errormessage = " ALTRO Payload Error ";
            break;
          case AltroErrType::ALTRO_MAPPING_ERROR:
            errormessage = " ALTRO Mapping Error ";
            break;
          case AltroErrType::CHANNEL_ERROR:
            errormessage = " Channel Error ";
            break;
          default:
            break;
        }
        if (mNumErrorMessages < mMaxErrorMessages) {
          LOG(ERROR) << " EMCAL raw task: " << errormessage << " in Supermodule " << feeID << std::endl;
          mNumErrorMessages++;
          if (mNumErrorMessages == mMaxErrorMessages) {
            LOG(ERROR) << "Max. amount of error messages (" << mMaxErrorMessages << " reached, further messages will be suppressed";
          }
        } else {
          mErrorMessagesSuppressed++;
        }
        //fill histograms  with error types
        mOutputDecoderErrors.push_back(errornum);
        continue;
      }

      LOG(DEBUG) << decoder.getRCUTrailer();
      // Apply zero suppression only in case it was enabled
      mRawFitter->setIsZeroSuppressed(decoder.getRCUTrailer().hasZeroSuppression());

      const auto& map = mMapper->getMappingForDDL(feeID);
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

        auto [phishift, etashift] = mGeometry->ShiftOnlineToOfflineCellIndexes(iSM, iRow, iCol);
        int CellID = mGeometry->GetAbsCellIdFromCellIndexes(iSM, phishift, etashift);

        // define the conatiner for the fit results, and perform the raw fitting using the stadnard raw fitter
        CaloFitResults fitResults;
        try {
          fitResults = mRawFitter->evaluate(chan.getBunches(), 0, 0);
          // Prevent negative entries - we should no longer get here as the raw fit usually will end in an error state
          if (fitResults.getAmp() < 0) {
            fitResults.setAmp(0.);
          }
          if (fitResults.getTime() < 0) {
            fitResults.setTime(0.);
          }
        } catch (CaloRawFitter::RawFitterError_t& fiterror) {
          if (mNumErrorMessages < mMaxErrorMessages) {
            LOG(ERROR) << "Failure in raw fitting: " << CaloRawFitter::createErrorMessage(fiterror);
            mNumErrorMessages++;
            if (mNumErrorMessages == mMaxErrorMessages) {
              LOG(ERROR) << "Max. amount of error messages (" << mMaxErrorMessages << " reached, further messages will be suppressed";
            }
          } else {
            mErrorMessagesSuppressed++;
          }
          mOutputDecoderErrors.emplace_back(feeID, -1, CaloRawFitter::getErrorNumber(fiterror));
        }
        currentCellContainer->emplace_back(CellID, fitResults.getAmp() * CONVADCGEV, fitResults.getTime(), chantype);
      }
    }
  }

  // Loop over BCs, sort cells with increasing tower ID and write to output containers
  for (auto [bc, cells] : cellBuffer) {
    mOutputTriggerRecords.emplace_back(bc, triggerBuffer[bc], mOutputCells.size(), cells->size());
    if (cells->size()) {
      // Sort cells according to cell ID
      std::sort(cells->begin(), cells->end(), [](Cell& lhs, Cell& rhs) { return lhs.getTower() < rhs.getTower(); });
      for (auto cell : *cells) {
        mOutputCells.push_back(cell);
      }
    }
  }

  LOG(DEBUG) << "[EMCALRawToCellConverter - run] Writing " << mOutputCells.size() << " cells ...";
  sendData(ctx, mOutputCells, mOutputTriggerRecords, mOutputDecoderErrors);
}

bool RawToCellConverterSpec::isLostTimeframe(framework::ProcessingContext& ctx) const
{
  constexpr auto originEMC = header::gDataOriginEMC;
  o2::framework::InputSpec dummy{"dummy",
                                 framework::ConcreteDataMatcher{originEMC,
                                                                header::gDataDescriptionRawData,
                                                                0xDEADBEEF}};
  for (const auto& ref : o2::framework::InputRecordWalker(ctx.inputs(), {dummy})) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    if (dh->payloadSize == 0) {
      return true;
    }
  }
  return false;
}

void RawToCellConverterSpec::sendData(framework::ProcessingContext& ctx, const std::vector<o2::emcal::Cell>& cells, const std::vector<o2::emcal::TriggerRecord>& triggers, const std::vector<ErrorTypeFEE>& decodingErrors) const
{
  constexpr auto originEMC = o2::header::gDataOriginEMC;
  ctx.outputs().snapshot(framework::Output{originEMC, "CELLS", mSubspecification, framework::Lifetime::Timeframe}, cells);
  ctx.outputs().snapshot(framework::Output{originEMC, "CELLSTRGR", mSubspecification, framework::Lifetime::Timeframe}, triggers);
  ctx.outputs().snapshot(framework::Output{originEMC, "DECODERERR", mSubspecification, framework::Lifetime::Timeframe}, decodingErrors);
}

o2::framework::DataProcessorSpec o2::emcal::reco_workflow::getRawToCellConverterSpec(bool askDISTSTF, int subspecification)
{
  constexpr auto originEMC = o2::header::gDataOriginEMC;
  std::vector<o2::framework::OutputSpec> outputs;

  outputs.emplace_back(originEMC, "CELLS", subspecification, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(originEMC, "CELLSTRGR", subspecification, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(originEMC, "DECODERERR", subspecification, o2::framework::Lifetime::Timeframe);

  std::vector<o2::framework::InputSpec> inputs{{"stf", o2::framework::ConcreteDataTypeMatcher{originEMC, o2::header::gDataDescriptionRawData}, o2::framework::Lifetime::Optional}};
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }

  return o2::framework::DataProcessorSpec{"EMCALRawToCellConverterSpec",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::emcal::reco_workflow::RawToCellConverterSpec>(subspecification),
                                          o2::framework::Options{
                                            {"fitmethod", o2::framework::VariantType::String, "gamma2", {"Fit method (standard or gamma2)"}},
                                            {"maxmessage", o2::framework::VariantType::Int, 100, {"Max. amout of error messages to be displayed"}}}};
}
