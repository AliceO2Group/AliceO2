// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "BareGBTDecoder.h"
#include "DetectorsRaw/RDHUtils.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/PageDecoder.h"
#include "MCHRawElecMap/Mapper.h"
#include "UserLogicEndpointDecoder.h"
#include <iostream>

namespace o2::mch::raw
{
namespace impl
{

template <typename FORMAT, typename CHARGESUM, int VERSION = 0>
struct PayloadDecoderImpl {

  using type = struct PayloadDecoderImplReturnType {
    void process(uint32_t, gsl::span<const std::byte>);
  };

  type operator()(const FeeLinkId& feeLinkId, DecodedDataHandlers decodedDataHandlers, FeeLink2SolarMapper fee2solar);
};

template <typename CHARGESUM, int VERSION>
struct PayloadDecoderImpl<UserLogicFormat, CHARGESUM, VERSION> {
  using type = UserLogicEndpointDecoder<CHARGESUM, VERSION>;

  type operator()(const FeeLinkId& feeLinkId, DecodedDataHandlers decodedDataHandlers, FeeLink2SolarMapper fee2solar)
  {
    return std::move(UserLogicEndpointDecoder<CHARGESUM, VERSION>(feeLinkId.feeId(), fee2solar, decodedDataHandlers));
  }
};

template <typename CHARGESUM>
struct PayloadDecoderImpl<BareFormat, CHARGESUM, 0> {
  using type = BareGBTDecoder<CHARGESUM>;

  type operator()(const FeeLinkId& feeLinkId, DecodedDataHandlers decodedDataHandlers, FeeLink2SolarMapper fee2solar)
  {
    auto solarId = fee2solar(feeLinkId);
    if (!solarId.has_value()) {
      throw std::logic_error(fmt::format("{} could not get solarId from feelinkid={}\n", __PRETTY_FUNCTION__, feeLinkId));
    }
    return std::move(BareGBTDecoder<CHARGESUM>(solarId.value(), decodedDataHandlers));
  }
};

template <typename FORMAT, typename CHARGESUM, int VERSION = 0>
class PageDecoderImpl
{
 public:
  PageDecoderImpl(DecodedDataHandlers decodedDataHandlers, FeeLink2SolarMapper fee2solar) : mDecodedDataHandlers{decodedDataHandlers},
                                                                                            mFee2SolarMapper(fee2solar)
  {
  }

  void operator()(Page page)
  {
    const void* rdhP = reinterpret_cast<const void*>(page.data());
    if (!o2::raw::RDHUtils::checkRDH(rdhP, true)) {
      throw std::invalid_argument("page does not start with a valid RDH");
    }

    auto feeId = o2::raw::RDHUtils::getFEEID(rdhP);
    auto linkId = o2::raw::RDHUtils::getLinkID(rdhP);
    FEEID f{feeId};
    FeeLinkId feeLinkId(f.id, linkId);

    auto p = mPayloadDecoders.find(feeLinkId);
    if (p == mPayloadDecoders.end()) {
      mPayloadDecoders.emplace(feeLinkId, PayloadDecoderImpl<FORMAT, CHARGESUM, VERSION>()(feeLinkId, mDecodedDataHandlers, mFee2SolarMapper));
      p = mPayloadDecoders.find(feeLinkId);
    }

    uint32_t orbit = o2::raw::RDHUtils::getHeartBeatOrbit(rdhP);
    auto rdhSize = o2::raw::RDHUtils::getHeaderSize(rdhP);
    auto payloadSize = o2::raw::RDHUtils::getMemorySize(rdhP) - rdhSize;
    // skip empty payloads, otherwise the orbit jumps are not correctly detected
    if (payloadSize > 0) {
      p->second.process(orbit, page.subspan(rdhSize, payloadSize));
    }
  }

 private:
  DecodedDataHandlers mDecodedDataHandlers;
  FeeLink2SolarMapper mFee2SolarMapper;
  std::map<FeeLinkId, typename PayloadDecoderImpl<FORMAT, CHARGESUM, VERSION>::type> mPayloadDecoders;
};

} // namespace impl

PageDecoder createPageDecoder(RawBuffer rdhBuffer, DecodedDataHandlers decodedDataHandlers, FeeLink2SolarMapper fee2solar)
{
  const void* rdhP = reinterpret_cast<const void*>(rdhBuffer.data());
  bool ok = o2::raw::RDHUtils::checkRDH(rdhP, true);
  if (!ok) {
    throw std::invalid_argument("rdhBuffer does not point to a valid RDH !");
  }
  auto linkId = o2::raw::RDHUtils::getLinkID(rdhP);
  auto feeId = o2::raw::RDHUtils::getFEEID(rdhP);
  FEEID f{feeId};
  if (linkId == 15) {
    if (f.ulFormatVersion != 0 && f.ulFormatVersion != 1) {
      throw std::logic_error(fmt::format("ulFormatVersion {} is unknown", static_cast<int>(f.ulFormatVersion)));
    }
    if (f.chargeSum) {
      if (f.ulFormatVersion == 0) {
        return impl::PageDecoderImpl<UserLogicFormat, ChargeSumMode, 0>(decodedDataHandlers, fee2solar);
      } else if (f.ulFormatVersion == 1) {
        return impl::PageDecoderImpl<UserLogicFormat, ChargeSumMode, 1>(decodedDataHandlers, fee2solar);
      }
    } else {
      if (f.ulFormatVersion == 0) {
        return impl::PageDecoderImpl<UserLogicFormat, SampleMode, 0>(decodedDataHandlers, fee2solar);
      } else if (f.ulFormatVersion == 1) {
        return impl::PageDecoderImpl<UserLogicFormat, SampleMode, 1>(decodedDataHandlers, fee2solar);
      }
    }
  } else {
    if (f.chargeSum) {
      return impl::PageDecoderImpl<BareFormat, ChargeSumMode>(decodedDataHandlers, fee2solar);
    } else {
      return impl::PageDecoderImpl<BareFormat, SampleMode>(decodedDataHandlers, fee2solar);
    }
  }
  return nullptr;
}

PageDecoder createPageDecoder(RawBuffer rdhBuffer, DecodedDataHandlers decodedDataHandlers)
{
  auto fee2solar = createFeeLink2SolarMapper<ElectronicMapperGenerated>();
  return createPageDecoder(rdhBuffer, decodedDataHandlers, fee2solar);
}

PageDecoder createPageDecoder(RawBuffer rdhBuffer, SampaChannelHandler channelHandler)
{
  DecodedDataHandlers handlers;
  handlers.sampaChannelHandler = channelHandler;
  return createPageDecoder(rdhBuffer, handlers);
}

PageDecoder createPageDecoder(RawBuffer rdhBuffer, SampaChannelHandler channelHandler, FeeLink2SolarMapper fee2solar)
{
  DecodedDataHandlers handlers;
  handlers.sampaChannelHandler = channelHandler;
  return createPageDecoder(rdhBuffer, handlers, fee2solar);
}

PageParser createPageParser()
{
  return [](RawBuffer buffer, PageDecoder pageDecoder) {
    size_t pos{0};
    const void* rdhP = reinterpret_cast<const void*>(buffer.data());
    bool ok = o2::raw::RDHUtils::checkRDH(rdhP, true);
    if (!ok) {
      throw std::invalid_argument("buffer does not start with a valid RDH !");
    }
    auto rdhSize = o2::raw::RDHUtils::getHeaderSize(rdhP);
    while (pos < buffer.size_bytes() - rdhSize) {
      const void* rdhP = reinterpret_cast<const void*>(buffer.data() + pos);
      bool ok = o2::raw::RDHUtils::checkRDH(rdhP, true);
      if (!ok) {
        throw std::invalid_argument(fmt::format("buffer at pos {} does not point to a valid RDH !", pos));
      }
      auto payloadSize = o2::raw::RDHUtils::getMemorySize(rdhP) - rdhSize;
      pageDecoder(buffer.subspan(pos, rdhSize + payloadSize));
      pos += o2::raw::RDHUtils::getOffsetToNext(rdhP);
    }
  };
}

} // namespace o2::mch::raw
