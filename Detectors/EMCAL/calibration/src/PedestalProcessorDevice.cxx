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

#include "CommonUtils/VerbosityConfig.h"
#include "DetectorsRaw/RDHUtils.h"
#include "EMCALBase/Geometry.h"
#include "EMCALCalibration/PedestalProcessorDevice.h"
#include "EMCALReconstruction/AltroDecoder.h"
#include "EMCALReconstruction/RawReaderMemory.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/InputRecordWalker.h"

#include <fairlogger/Logger.h>

using namespace o2::emcal;

void PedestalProcessorDevice::init(o2::framework::InitContext& ctx)
{
  LOG(debug) << "[EMCALRawToCellConverter - init] Initialize converter ";
  if (!mGeometry) {
    mGeometry = Geometry::GetInstanceFromRunNumber(300000);
  }
  if (!mGeometry) {
    LOG(error) << "Failure accessing geometry";
  }

  if (!mMapper) {
    mMapper = std::unique_ptr<MappingHandler>(new o2::emcal::MappingHandler);
  }
  if (!mMapper) {
    LOG(error) << "Failed to initialize mapper";
  }
}

void PedestalProcessorDevice::run(o2::framework::ProcessingContext& ctx)
{
  constexpr auto originEMC = o2::header::gDataOriginEMC;
  constexpr auto descRaw = o2::header::gDataDescriptionRawData;

  mPedestalData.reset();

  if (isLostTimeframe(ctx)) {
    sendData(ctx, mPedestalData);
    return;
  }

  std::vector<framework::InputSpec> filter{{"filter", framework::ConcreteDataTypeMatcher(originEMC, descRaw)}};
  for (const auto& rawData : framework::InputRecordWalker(ctx.inputs(), filter)) {
    // Skip SOX headers
    auto rdhblock = reinterpret_cast<const o2::header::RDHAny*>(rawData.payload);
    if (o2::raw::RDHUtils::getHeaderSize(rdhblock) == static_cast<int>(o2::framework::DataRefUtils::getPayloadSize(rawData))) {
      continue;
    }

    o2::emcal::RawReaderMemory rawreader(framework::DataRefUtils::as<const char>(rawData));
    rawreader.setRangeSRUDDLs(0, 39);

    // loop over all the DMA pages
    while (rawreader.hasNext()) {
      try {
        rawreader.next();
      } catch (RawDecodingError& e) {
        LOG(error) << e.what();
        if (e.getErrorType() == RawDecodingError::ErrorType_t::HEADER_DECODING || e.getErrorType() == RawDecodingError::ErrorType_t::HEADER_INVALID) {
          // We must break in case of header decoding as the offset to the next payload is lost
          // consequently the parser does not know where to continue leading to an infinity loop
          break;
        }
        // We must skip the page as payload is not consistent
        // otherwise the next functions will rethrow the exceptions as
        // the page format does not follow the expected format
        continue;
      }
      for (const auto& e : rawreader.getMinorErrors()) {
        LOG(warning) << "Found minor Raw decoder error in FEE " << e.getFEEID() << ": " << RawDecodingError::getErrorCodeTitles(RawDecodingError::ErrorTypeToInt(e.getErrorType()));
      }

      auto& header = rawreader.getRawHeader();
      auto feeID = raw::RDHUtils::getFEEID(header);
      auto triggerbits = raw::RDHUtils::getTriggerType(header);

      if (feeID >= 40) {
        continue; // skip STU ddl
      }

      // use the altro decoder to decode the raw data, and extract the RCU trailer
      AltroDecoder decoder(rawreader);
      // check the words of the payload exception in altrodecoder
      try {
        decoder.decode();
      } catch (AltroDecoderError& e) {
        LOG(error) << e.what();
        continue;
      }
      for (const auto& e : decoder.getMinorDecodingErrors()) {
        LOG(warning) << e.what();
      }

      try {

        const auto& map = mMapper->getMappingForDDL(feeID);
        uint16_t iSM = feeID / 2;

        // Loop over all the channels
        int nBunchesNotOK = 0;
        for (auto& chan : decoder.getChannels()) {
          int iRow, iCol;
          ChannelType_t chantype;
          try {
            iRow = map.getRow(chan.getHardwareAddress());
            iCol = map.getColumn(chan.getHardwareAddress());
            chantype = map.getChannelType(chan.getHardwareAddress());
          } catch (Mapper::AddressNotFoundException& ex) {
            LOG(error) << ex.what();
            continue;
          }

          if (!(chantype == o2::emcal::ChannelType_t::HIGH_GAIN || chantype == o2::emcal::ChannelType_t::LOW_GAIN || chantype == o2::emcal::ChannelType_t::LEDMON)) {
            continue;
          }

          int CellID = -1;
          bool isLowGain = false;
          try {
            if (chantype == o2::emcal::ChannelType_t::HIGH_GAIN || chantype == o2::emcal::ChannelType_t::LOW_GAIN) {
              // high- / low-gain cell
              CellID = getCellAbsID(iSM, iCol, iRow);
              isLowGain = chantype == o2::emcal::ChannelType_t::LOW_GAIN;
            } else {
              CellID = geLEDMONAbsID(iSM, iCol); // Module index encoded in colum for LEDMONs
              isLowGain = iRow == 0;             // For LEDMONs gain type is encoded in the row (0 - low gain, 1 - high gain)
            }
          } catch (ModuleIndexException& e) {
            LOG(error) << e.what();
            continue;
          }

          // Fill pedestal object
          for (auto& bunch : chan.getBunches()) {
            for (auto e : bunch.getADC()) {
              mPedestalData.fillADC(e, CellID, isLowGain, chantype == o2::emcal::ChannelType_t::LEDMON);
            }
          }
        }
      } catch (o2::emcal::MappingHandler::DDLInvalid& ddlerror) {
        // Unable to catch mapping
        LOG(error) << ddlerror.what();
      }
    }
  }

  sendData(ctx, mPedestalData);
}

int PedestalProcessorDevice::getCellAbsID(int supermoduleID, int column, int row) const
{
  auto [phishift, etashift] = mGeometry->ShiftOnlineToOfflineCellIndexes(supermoduleID, row, column);
  int cellID = mGeometry->GetAbsCellIdFromCellIndexes(supermoduleID, phishift, etashift);
  if (cellID > 17664 || cellID < 0) {
    throw ModuleIndexException(cellID, column, row, etashift, phishift);
  }
  return cellID;
}

int PedestalProcessorDevice::geLEDMONAbsID(int supermoduleID, int moduleID) const
{
  if (moduleID >= o2::emcal::EMCAL_LEDREFS || moduleID < 0) {
    throw ModuleIndexException(moduleID);
  }
  return supermoduleID * o2::emcal::EMCAL_LEDREFS + moduleID;
}

bool PedestalProcessorDevice::isLostTimeframe(framework::ProcessingContext& ctx) const
{
  constexpr auto originEMC = header::gDataOriginEMC;
  o2::framework::InputSpec dummy{"dummy",
                                 framework::ConcreteDataMatcher{originEMC,
                                                                header::gDataDescriptionRawData,
                                                                0xDEADBEEF}};
  static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
  for (const auto& ref : o2::framework::InputRecordWalker(ctx.inputs(), {dummy})) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(ref);
    if (payloadSize == 0) {
      auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
      if (++contDeadBeef <= maxWarn) {
        LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
             contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
      }
      return true;
    }
  }
  contDeadBeef = 0; // if good data, reset the counter
  return false;
}

void PedestalProcessorDevice::sendData(framework::ProcessingContext& ctx, const PedestalProcessorData& data) const
{
  constexpr auto originEMC = o2::header::gDataOriginEMC;
  ctx.outputs().snapshot(framework::Output{originEMC, "PEDDATA", 0}, data);
}

o2::framework::DataProcessorSpec o2::emcal::getPedestalProcessorDevice(bool askDistSTF)
{
  constexpr auto originEMC = o2::header::gDataOriginEMC;
  std::vector<o2::framework::InputSpec> inputs{{"stf", o2::framework::ConcreteDataTypeMatcher{originEMC, o2::header::gDataDescriptionRawData}, o2::framework::Lifetime::Timeframe}};
  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back(originEMC, "PEDDATA", 0, o2::framework::Lifetime::Timeframe);
  if (askDistSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe);
  }

  return o2::framework::DataProcessorSpec{
    "PedestalProcessor",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<PedestalProcessorDevice>()},
    o2::framework::Options{}};
}