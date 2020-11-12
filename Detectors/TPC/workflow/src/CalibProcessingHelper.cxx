// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ConcreteDataMatcher.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"

#include "DPLUtils/RawParser.h"
#include "TPCBase/RDHUtils.h"
#include "TPCReconstruction/RawReaderCRU.h"

#include "TPCWorkflow/CalibProcessingHelper.h"

using namespace o2::tpc;
using namespace o2::framework;

void calib_processing_helper::processRawData(o2::framework::InputRecord& inputs, std::unique_ptr<RawReaderCRU>& reader, bool useOldSubspec)
{
  std::vector<InputSpec> filter = {{"check", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "RAWDATA"}, Lifetime::Timeframe}};

  for (auto const& ref : InputRecordWalker(inputs, filter)) {
    const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);

    // ---| extract hardware information to do the processing |---
    const auto subSpecification = dh->subSpecification;
    rdh_utils::FEEIDType feeID = (rdh_utils::FEEIDType)dh->subSpecification;
    rdh_utils::FEEIDType cruID, linkID, endPoint;

    if (useOldSubspec) {
      //---| old definition by Gvozden |---
      cruID = (rdh_utils::FEEIDType)(subSpecification >> 16);
      linkID = (rdh_utils::FEEIDType)((subSpecification + (subSpecification >> 8)) & 0xFF) - 1;
      endPoint = (rdh_utils::FEEIDType)((subSpecification >> 8) & 0xFF) > 0;
    } else {
      //---| new definition by David |---
      rdh_utils::getMapping(feeID, cruID, endPoint, linkID);
    }

    const auto globalLinkID = linkID + endPoint * 12;

    // ---| update hardware information in the reader |---
    reader->forceCRU(cruID);
    reader->setLink(globalLinkID);

    LOGP(info, "Specifier: {}/{}/{}", dh->dataOrigin.as<std::string>(), dh->dataDescription.as<std::string>(), subSpecification);
    LOGP(info, "Payload size: {}", dh->payloadSize);
    LOGP(info, "CRU: {}; linkID: {}; endPoint: {}; globalLinkID: {}", cruID, linkID, endPoint, globalLinkID);

    // TODO: exception handling needed?
    const gsl::span<const char> raw = inputs.get<gsl::span<char>>(ref);
    o2::framework::RawParser parser(raw.data(), raw.size());

    // TODO: it would be better to have external event handling and then moving the event processing functionality to CalibRawBase and RawReader to not repeat it in other places
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
        gFrame.readFromMemory(gsl::span<const o2::byte>(data + i, 16));

        // extract the half words from the 4 32-bit words
        gFrame.getFrameHalfWords();

        gFrame.getAdcValues(rawData);
        gFrame.updateSyncCheck(false);

        ++iFrame;
      }
    }

    reader->runADCDataCallback(rawData);
  }
}
