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

#include <fmt/core.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <fmt/chrono.h>

#include "GPUO2Interface.h"
#include "GPUParam.h"
#include "GPUReconstructionConvert.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "DPLUtils/RawParser.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Headers/DataHeaderHelpers.h"
#include "Headers/RDHAny.h"
#include "DataFormatsTPC/ZeroSuppressionLinkBased.h"
#include "DataFormatsTPC/RawDataTypes.h"
#include "DataFormatsTPC/Digit.h"

#include "TPCBase/RDHUtils.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCReconstruction/RawProcessingHelpers.h"

#include "TPCWorkflow/CalibProcessingHelper.h"

using namespace o2::tpc;
using namespace o2::framework;
using RDHUtils = o2::raw::RDHUtils;

void processGBT(o2::framework::RawParser<>& parser, std::unique_ptr<RawReaderCRU>& reader, const rdh_utils::FEEIDType feeID);
void processLinkZS(o2::framework::RawParser<>& parser, std::unique_ptr<RawReaderCRU>& reader, uint32_t firstOrbit, uint32_t syncOffsetReference, uint32_t decoderType);
uint32_t getBCsyncOffsetReference(InputRecord& inputs, const std::vector<InputSpec>& filter);

uint64_t calib_processing_helper::processRawData(o2::framework::InputRecord& inputs, std::unique_ptr<RawReaderCRU>& reader, bool useOldSubspec, const std::vector<int>& sectors, size_t* nerrors, uint32_t syncOffsetReference, uint32_t decoderType)
{
  std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}};
  size_t errorCount = 0;
  // TODO: check if presence of data sampling can be checked in another way
  bool sampledData = true;
  for ([[maybe_unused]] auto const& ref : InputRecordWalker(inputs, filter)) {
    sampledData = false;
    break;
  }
  // used for online monitor
  if (sampledData) {
    filter = {{"sampled-rawdata", ConcreteDataTypeMatcher{"DS2", "RAWDATA"}, Lifetime::Timeframe}};
    for ([[maybe_unused]] auto const& ref : InputRecordWalker(inputs, filter)) {
      sampledData = false;
      break;
    }
  }
  // used for QC
  if (sampledData) {
    filter = {{"sampled-rawdata", ConcreteDataTypeMatcher{"DS", "RAWDATA"}, Lifetime::Timeframe}};
    LOGP(info, "Using sampled data");
  }

  uint64_t activeSectors = 0;
  uint32_t firstOrbit = 0;
  bool readFirst = false;
  bool readFirstZS = false;

  // for LinkZS data the maximum sync offset is needed to align the data properly.
  // getBCsyncOffsetReference only works, if the full TF is seen. Alternatively, this value could be set
  // fixed to e.g. 144 or 152 which is the maximum sync delay expected
  // this is less precise and might lead to more time bins which have to be removed at the beginnig
  // or end of the TF
  // uint32_t syncOffsetReference = getBCsyncOffsetReference(inputs, filter);
  // uint32_t syncOffsetReference = 144;

  for (auto const& ref : InputRecordWalker(inputs, filter)) {
    const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto payloadSize = DataRefUtils::getPayloadSize(ref);
    // skip empty HBF
    if (payloadSize == 2 * sizeof(o2::header::RAWDataHeader)) {
      continue;
    }

    firstOrbit = dh->firstTForbit;

    // ---| extract hardware information to do the processing |---
    const auto subSpecification = dh->subSpecification;
    rdh_utils::FEEIDType feeID = (rdh_utils::FEEIDType)dh->subSpecification;

    if (useOldSubspec) {
      //---| old definition by Gvozden |---
      // TODO: make auto detect from firt RDH?
      const auto cruID = (rdh_utils::FEEIDType)(subSpecification >> 16);
      const auto linkID = (rdh_utils::FEEIDType)((subSpecification + (subSpecification >> 8)) & 0xFF) - 1;
      const auto endPoint = (rdh_utils::FEEIDType)((subSpecification >> 8) & 0xFF) > 0;
      feeID = rdh_utils::getFEEID(cruID, endPoint, linkID);
    }

    const uint64_t sector = rdh_utils::getCRU(feeID) / 10;

    // sector selection should be better done by directly subscribing to a range of subspecs. But this might not be that simple
    if (sectors.size() && (std::find(sectors.begin(), sectors.end(), int(sector)) == sectors.end())) {
      continue;
    }

    activeSectors |= (0x1 << sector);

    // ===| for debugging only |===
    // remove later
    rdh_utils::FEEIDType cruID, linkID, endPoint;
    rdh_utils::getMapping(feeID, cruID, endPoint, linkID);
    const auto globalLinkID = linkID + endPoint * 12;
    LOGP(debug, "Specifier: {}/{}/{} Part {} of {}", dh->dataOrigin, dh->dataDescription, subSpecification, dh->splitPayloadIndex, dh->splitPayloadParts);
    LOGP(debug, "Payload size: {}", payloadSize);
    LOGP(debug, "CRU: {}; linkID: {}; endPoint: {}; globalLinkID: {}", cruID, linkID, endPoint, globalLinkID);
    // ^^^^^^

    // TODO: exception handling needed?
    const gsl::span<const char> raw = inputs.get<gsl::span<char>>(ref);
    std::unique_ptr<o2::framework::RawParser<8192>> rawparserPtr;
    try {

      o2::framework::RawParser parser(raw.data(), raw.size());
      // detect decoder type by analysing first RDH
      bool isLinkZS = false;
      {
        auto it = parser.begin();
        auto rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
        const auto rdhVersion = RDHUtils::getVersion(rdhPtr);
        if (!rdhPtr || rdhVersion < 6) {
          throw std::runtime_error(fmt::format("could not get RDH from packet, or version {} < 6", rdhVersion).data());
        }
        const auto link = RDHUtils::getLinkID(*rdhPtr);
        const auto detField = RDHUtils::getDetectorField(*rdhPtr);
        const auto feeID = RDHUtils::getFEEID(*rdhPtr);
        const auto feeLinkID = rdh_utils::getLink(feeID);
        if ((link == 0 || link == rdh_utils::UserLogicLinkID) && (detField == raw_data_types::LinkZS || ((feeLinkID == rdh_utils::ILBZSLinkID || feeLinkID == rdh_utils::DLBZSLinkID) && detField == raw_data_types::ZS))) {
          isLinkZS = true;
          if (!readFirstZS) {
            if (feeLinkID == rdh_utils::DLBZSLinkID) {
              LOGP(info, "Detected Dense Link-based zero suppression");
            } else if (feeLinkID == rdh_utils::ILBZSLinkID) {
              LOGP(info, "Detected Improved Link-based zero suppression");
            } else {
              LOGP(info, "Detected Link-based zero suppression");
            }
            if (!reader->getManager() || !reader->getManager()->getLinkZSCallback()) {
              LOGP(fatal, "LinkZSCallback must be set in RawReaderCRUManager");
            }
            readFirstZS = true;
          }
        }

        // firstOrbit = RDHUtils::getHeartBeatOrbit(*rdhPtr);
        if (!readFirst) {
          LOGP(info, "First orbit in present TF: {}", firstOrbit);
        }
        readFirst = true;
      }

      if (isLinkZS) {
        processLinkZS(parser, reader, firstOrbit, syncOffsetReference, decoderType);
      } else {
        processGBT(parser, reader, feeID);
      }

    } catch (const std::exception& e) {
      // error message throtteling
      using namespace std::literals::chrono_literals;
      static std::unordered_map<uint32_t, size_t> nErrorPerSubspec;
      static std::chrono::time_point<std::chrono::steady_clock> lastReport = std::chrono::steady_clock::now();
      const auto now = std::chrono::steady_clock::now();
      static size_t reportedErrors = 0;
      const size_t MAXERRORS = 10;
      const auto sleepTime = 10min;
      ++nErrorPerSubspec[subSpecification];

      if ((now - lastReport) < sleepTime) {
        if (reportedErrors < MAXERRORS) {
          ++reportedErrors;
          std::string sleepInfo;
          if (reportedErrors == MAXERRORS) {
            sleepInfo = fmt::format(", maximum error count ({}) reached, not reporting for the next {}", MAXERRORS, sleepTime);
          }
          LOGP(alarm, "EXCEPTIION in processRawData: {} -> skipping part:{}/{} of spec:{}/{}/{}, size:{}, error count for subspec: {}{}", e.what(), dh->splitPayloadIndex, dh->splitPayloadParts,
               dh->dataOrigin, dh->dataDescription, subSpecification, payloadSize, nErrorPerSubspec.at(subSpecification), sleepInfo);
          lastReport = now;
        }
      } else {
        lastReport = now;
        reportedErrors = 0;
      }
      errorCount++;
      continue;
    }
  }
  if (nerrors) {
    *nerrors += errorCount;
  }
  return activeSectors;
}

void processGBT(o2::framework::RawParser<>& parser, std::unique_ptr<RawReaderCRU>& reader, const rdh_utils::FEEIDType feeID)
{
  // TODO: currently this will only work for HBa1, since the sync is in the first packet and
  // the decoder expects all packets of one link to be processed at once
  rdh_utils::FEEIDType cruID, linkID, endPoint;
  rdh_utils::getMapping(feeID, cruID, endPoint, linkID);
  const auto globalLinkID = linkID + endPoint * 12;

  // ---| update hardware information in the reader |---
  reader->forceCRU(cruID);
  reader->setLink(globalLinkID);

  rawreader::ADCRawData rawData;
  rawreader::GBTFrame gFrame;

  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {

    const auto size = it.size();
    auto data = it.data();
    // LOGP(info, "Data size: {}", size);

    int iFrame = 0;
    for (int i = 0; i < size; i += 16) {
      gFrame.setFrameNumber(iFrame);
      gFrame.setPacketNumber(iFrame / 508);
      gFrame.readFromMemory(gsl::span<const std::byte>((std::byte*)data + i, 16));

      // extract the half words from the 4 32-bit words
      gFrame.getFrameHalfWords();

      gFrame.getAdcValues(rawData);
      gFrame.updateSyncCheck(false);

      ++iFrame;
    }
  }

  reader->runADCDataCallback(rawData);
}

void processLinkZS(o2::framework::RawParser<>& parser, std::unique_ptr<RawReaderCRU>& reader, uint32_t firstOrbit, uint32_t syncOffsetReference, uint32_t decoderType)
{
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    auto rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
    const auto rdhVersion = RDHUtils::getVersion(rdhPtr);
    if (!rdhPtr || rdhVersion < 6) {
      throw std::runtime_error(fmt::format("could not get RDH from packet, or version {} < 6", rdhVersion).data());
    }
    // workaround for MW2 data
    // const bool useTimeBins = true;
    // const auto cru = RDHUtils::getCRUID(*rdhPtr);
    // const auto feeID = (RDHUtils::getFEEID(*rdhPtr) & 0x7f) | (cru << 7);

    // skip all data that is not Link-base zero suppression
    const auto link = RDHUtils::getLinkID(*rdhPtr);
    const auto detField = RDHUtils::getDetectorField(*rdhPtr);
    const auto feeID = RDHUtils::getFEEID(*rdhPtr);
    const auto linkID = rdh_utils::getLink(feeID);
    if (!((detField == raw_data_types::LinkZS) ||
          ((detField == raw_data_types::RAWDATA || detField == 0xdeadbeef) && (link == rdh_utils::UserLogicLinkID)) ||
          ((linkID == rdh_utils::ILBZSLinkID || linkID == rdh_utils::DLBZSLinkID) && (detField == raw_data_types::Type::ZS)))) {
      continue;
    }

    if ((decoderType == 1) && (linkID == rdh_utils::ILBZSLinkID || linkID == rdh_utils::DLBZSLinkID) && (detField == raw_data_types::Type::ZS)) {
      std::vector<Digit> digits;
      static o2::gpu::GPUParam gpuParam;
      static o2::gpu::GPUReconstructionZSDecoder gpuDecoder;
      gpuDecoder.DecodePage(digits, (const void*)it.raw(), firstOrbit, gpuParam);
      for (const auto& digit : digits) {
        reader->getManager()->getLinkZSCallback()(digit.getCRU(), digit.getRow(), digit.getPad(), digit.getTimeStamp(), digit.getChargeFloat());
      }
    } else {
      const auto orbit = RDHUtils::getHeartBeatOrbit(*rdhPtr);
      const auto data = (const char*)it.data();
      const auto size = it.size();
      raw_processing_helpers::processZSdata(data, size, feeID, orbit, firstOrbit, syncOffsetReference, reader->getManager()->getLinkZSCallback());
    }
  }
}

// find the global sync offset reference, using the large sync offset to avoid negative time bins
uint32_t getBCsyncOffsetReference(InputRecord& inputs, const std::vector<InputSpec>& filter)
{
  uint32_t syncOffsetReference = 144;

  for (auto const& ref : InputRecordWalker(inputs, filter)) {
    const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto payloadSize = DataRefUtils::getPayloadSize(ref);
    // skip empty HBF
    if (payloadSize == 2 * sizeof(o2::header::RAWDataHeader)) {
      continue;
    }

    const gsl::span<const char> raw = inputs.get<gsl::span<char>>(ref);
    o2::framework::RawParser parser(raw.data(), raw.size());

    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      auto rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
      const auto rdhVersion = RDHUtils::getVersion(rdhPtr);
      if (!rdhPtr || rdhVersion < 6) {
        throw std::runtime_error(fmt::format("could not get RDH from packet, or version {} < 6", rdhVersion).data());
      }

      // only process LinkZSdata, only supported for data where this is already set in the UL
      const auto link = RDHUtils::getLinkID(*rdhPtr);
      const auto detField = RDHUtils::getDetectorField(*rdhPtr);
      const auto feeID = RDHUtils::getFEEID(*rdhPtr);
      const auto linkID = rdh_utils::getLink(feeID);
      if (!((detField == raw_data_types::LinkZS) ||
            ((detField == raw_data_types::RAWDATA || detField == 0xdeadbeef) && (link == rdh_utils::UserLogicLinkID)) ||
            ((linkID == rdh_utils::ILBZSLinkID) && (detField == raw_data_types::Type::ZS)))) {
        continue;
      }

      const auto data = (const char*)it.data();
      const auto size = it.size();

      zerosupp_link_based::ContainerZS* zsdata = (zerosupp_link_based::ContainerZS*)data;
      const zerosupp_link_based::ContainerZS* const zsdataEnd = (zerosupp_link_based::ContainerZS*)(data + size);

      while (zsdata < zsdataEnd) {
        const auto& header = zsdata->cont.header;
        // align to header word if needed
        if (!header.hasCorrectMagicWord()) {
          zsdata = (zerosupp_link_based::ContainerZS*)((const char*)zsdata + sizeof(zerosupp_link_based::Header));
          continue;
        }

        // skip trigger info
        if (header.isTriggerInfo()) {
          zsdata = zsdata->next();
          continue;
        }

        syncOffsetReference = std::max(header.syncOffsetBC, syncOffsetReference);

        // only read first time bin for each link
        break;
      }
    }
  }

  LOGP(info, "syncOffsetReference in this TF: {}", syncOffsetReference);
  return syncOffsetReference;
}
