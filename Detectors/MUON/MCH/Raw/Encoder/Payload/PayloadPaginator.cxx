// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawEncoderPayload/PayloadPaginator.h"
#include <optional>
#include "MCHRawElecMap/Mapper.h"
#include "MCHRawCommon/DataFormats.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Headers/RDHAny.h"
#include <iostream>
#include "DetectorsRaw/RawFileWriter.h"
#include "MCHRawEncoderPayload/DataBlock.h"
#include "Framework/Logger.h"

namespace o2::mch::raw
{

uint64_t encode(const LinkInfo& li)
{
  return static_cast<uint64_t>(li.endPoint) |
         static_cast<uint64_t>(li.linkId) << 8 |
         static_cast<uint64_t>(li.cruId) << 24 |
         static_cast<uint64_t>(li.feeId) << 40;
}

bool operator<(const LinkInfo& li1, const LinkInfo& li2)
{
  return encode(li1) < encode(li2);
}

std::ostream& operator<<(std::ostream& os, const LinkInfo& li)
{
  os << fmt::format("FEEID {:6d} CRU {:4d} LINK {:4d} EP {:1d}",
                    li.feeId, li.cruId, li.linkId, li.endPoint);
  return os;
}

template <typename ELECMAP, typename FORMAT, typename CHARGESUM, int VERSION>
Solar2LinkInfo createSolar2LinkInfo()
{
  auto s2f = createSolar2FeeLinkMapper<ELECMAP>();
  return [s2f](uint16_t solarId) -> std::optional<LinkInfo> {
    auto f = s2f(solarId);
    if (!f.has_value()) {
      return std::nullopt;
    };
    LinkInfo li;
    li.endPoint = f->feeId() % 2;
    li.cruId = (f->feeId() - li.endPoint) / 2;
    li.feeId = f->feeId() | extraFeeIdChargeSumMask<CHARGESUM>() | extraFeeIdVersionMask<VERSION>();
    li.linkId = linkRemapping<FORMAT>(f->linkId());
    return li;
  };
}

o2::header::RDHAny rdhFromLinkInfo(LinkInfo li)
{
  o2::header::RDHAny rdh;
  o2::raw::RDHUtils::setFEEID(rdh, li.feeId);
  o2::raw::RDHUtils::setCRUID(rdh, li.cruId);
  o2::raw::RDHUtils::setEndPointID(rdh, li.endPoint);
  o2::raw::RDHUtils::setLinkID(rdh, li.linkId);
  return rdh;
}

void registerLinks(o2::raw::RawFileWriter& rawFileWriter,
                   std::string outputBase,
                   const std::set<LinkInfo>& links,
                   bool filePerLink)
{
  std::string output = fmt::format("{:s}.raw", outputBase);
  for (auto li : links) {
    if (filePerLink) {
      output = fmt::format("{:s}_feeid{:d}.raw", outputBase, li.feeId);
    }
    rawFileWriter.registerLink(rdhFromLinkInfo(li), output);
  }
}

void paginate(o2::raw::RawFileWriter& rawFileWriter,
              gsl::span<const std::byte> buffer,
              const std::set<LinkInfo>& links,
              Solar2LinkInfo solar2LinkInfo)
{
  std::set<DataBlockRef> dataBlockRefs;
  forEachDataBlockRef(
    buffer, [&](const DataBlockRef& ref) {
      dataBlockRefs.insert(ref);
    });

  std::map<o2::InteractionRecord, std::set<LinkInfo>> filled;

  // fill output buffer with actual data
  for (auto r : dataBlockRefs) {
    auto& b = r.block;
    auto& h = b.header;
    auto solarId = r.block.header.solarId;
    auto li = solar2LinkInfo(solarId);
    if (!li.has_value()) {
      throw std::runtime_error(fmt::format("Could not get fee,cru,link,endpoint for solarId {}", solarId));
    }
    filled[o2::InteractionRecord{h.bc, h.orbit}].insert(li.value());
    rawFileWriter.addData(rdhFromLinkInfo(li.value()),
                          {h.bc, h.orbit},
                          gsl::span<char>(const_cast<char*>(reinterpret_cast<const char*>(&b.payload[0])),
                                          b.payload.size()));
  }

  // loop over the used interaction records and ensure that we call
  // addData for each link where it was not already called.

  static std::array<char, 64> nothing;

  for (auto p : filled) {
    auto ir = p.first;
    auto& filledLinks = p.second;
    std::set<LinkInfo> addDataNotAlreadyCalled;
    std::set_difference(links.begin(), links.end(), filledLinks.begin(), filledLinks.end(),
                        std::inserter(addDataNotAlreadyCalled, addDataNotAlreadyCalled.end()));
    for (auto li : addDataNotAlreadyCalled) {
      rawFileWriter.addData(rdhFromLinkInfo(li), ir, nothing);
    }
  }
}

// instanciate only the specializations that actually make sense.
template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperGenerated, BareFormat, SampleMode, 0>();
template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperGenerated, BareFormat, ChargeSumMode, 0>();

template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperGenerated, UserLogicFormat, SampleMode, 0>();
template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperGenerated, UserLogicFormat, ChargeSumMode, 0>();
template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperGenerated, UserLogicFormat, SampleMode, 1>();
template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperGenerated, UserLogicFormat, ChargeSumMode, 1>();

template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperDummy, BareFormat, SampleMode, 0>();
template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperDummy, BareFormat, ChargeSumMode, 0>();

template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperDummy, UserLogicFormat, SampleMode, 0>();
template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperDummy, UserLogicFormat, ChargeSumMode, 0>();
template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperDummy, UserLogicFormat, SampleMode, 1>();
template Solar2LinkInfo createSolar2LinkInfo<ElectronicMapperDummy, UserLogicFormat, ChargeSumMode, 1>();

} // namespace o2::mch::raw
