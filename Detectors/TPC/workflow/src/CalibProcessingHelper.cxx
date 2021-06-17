// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <vector>
#include <algorithm>

#include "Framework/ConcreteDataMatcher.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "DPLUtils/RawParser.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Headers/DataHeaderHelpers.h"
#include "CommonConstants/LHCConstants.h"

#include "TPCBase/RDHUtils.h"
#include "TPCReconstruction/RawReaderCRU.h"
#include "TPCReconstruction/RawProcessingHelpers.h"

#include "TPCWorkflow/CalibProcessingHelper.h"

using namespace o2::tpc;
using namespace o2::framework;
using RDHUtils = o2::raw::RDHUtils;

void processGBT(o2::framework::RawParser<>& parser, std::unique_ptr<RawReaderCRU>& reader, const rdh_utils::FEEIDType feeID);
void processLinkZS(o2::framework::RawParser<>& parser, std::unique_ptr<RawReaderCRU>& reader, uint32_t firstOrbit);

uint64_t calib_processing_helper::processRawData(o2::framework::InputRecord& inputs, std::unique_ptr<RawReaderCRU>& reader, bool useOldSubspec, const std::vector<int>& sectors)
{
  std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}};

  // TODO: check if presence of data sampling can be checked in another way
  bool sampledData = true;
  for ([[maybe_unused]] auto const& ref : InputRecordWalker(inputs, filter)) {
    sampledData = false;
    break;
  }
  if (sampledData) {
    filter = {{"sampled-rawdata", ConcreteDataTypeMatcher{"DS", "RAWDATA"}, Lifetime::Timeframe}};
    LOGP(info, "Using sampled data");
  }

  uint64_t activeSectors = 0;
  bool isLinkZS = false;
  bool readFirst = false;
  uint32_t firstOrbit = 0;

  for (auto const& ref : InputRecordWalker(inputs, filter)) {
    const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
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
    LOGP(info, "Specifier: {}/{}/{}", dh->dataOrigin, dh->dataDescription, subSpecification);
    LOGP(info, "Payload size: {}", dh->payloadSize);
    LOGP(info, "CRU: {}; linkID: {}; endPoint: {}; globalLinkID: {}", cruID, linkID, endPoint, globalLinkID);
    // ^^^^^^

    // TODO: exception handling needed?
    const gsl::span<const char> raw = inputs.get<gsl::span<char>>(ref);
    o2::framework::RawParser parser(raw.data(), raw.size());

    // detect decoder type by analysing first RDH
    if (!readFirst) {
      auto it = parser.begin();
      auto* rdhPtr = it.get_if<o2::header::RAWDataHeaderV6>();
      if (!rdhPtr) {
        LOGP(fatal, "could not get RDH from packet");
      }
      const auto link = RDHUtils::getLinkID(*rdhPtr);
      if (link == rdh_utils::UserLogicLinkID) {
        LOGP(info, "Detected Link-based zero suppression");
        isLinkZS = true;
        if (!reader->getManager() || !reader->getManager()->getLinkZSCallback()) {
          LOGP(fatal, "LinkZSCallback must be set in RawReaderCRUManager");
        }
      }

      //firstOrbit = RDHUtils::getHeartBeatOrbit(*rdhPtr);
      LOGP(info, "First orbit in present TF: {}", firstOrbit);
      readFirst = true;
    }

    if (isLinkZS) {
      processLinkZS(parser, reader, firstOrbit);
    } else {
      processGBT(parser, reader, feeID);
    }
  }

  return activeSectors;
}

void processGBT(o2::framework::RawParser<>& parser, std::unique_ptr<RawReaderCRU>& reader, const rdh_utils::FEEIDType feeID)
{
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
    //LOGP(info, "Data size: {}", size);

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

void processLinkZS(o2::framework::RawParser<>& parser, std::unique_ptr<RawReaderCRU>& reader, uint32_t firstOrbit)
{
  for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
    auto* rdhPtr = it.get_if<o2::header::RAWDataHeaderV6>();
    if (!rdhPtr) {
      LOGP(fatal, "could not get RDH from packet");
    }
    // workaround for MW2 data
    //const bool useTimeBins = true;
    //const auto cru = RDHUtils::getCRUID(*rdhPtr);
    //const auto feeID = (RDHUtils::getFEEID(*rdhPtr) & 0x7f) | (cru << 7);

    const bool useTimeBins = false;
    const auto feeID = RDHUtils::getFEEID(*rdhPtr);
    const auto orbit = RDHUtils::getHeartBeatOrbit(*rdhPtr);
    const auto data = (const char*)it.data();
    const auto size = it.size();
    const auto globalBCOffset = (orbit - firstOrbit) * o2::constants::lhc::LHCMaxBunches;
    raw_processing_helpers::processZSdata(data, size, feeID, globalBCOffset, reader->getManager()->getLinkZSCallback(), useTimeBins);
  }
}
