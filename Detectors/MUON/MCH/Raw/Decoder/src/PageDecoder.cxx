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
#include "Headers/RAWDataHeader.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawCommon/RDHManip.h"
#include "MCHRawDecoder/PageDecoder.h"
#include "UserLogicEndpointDecoder.h"
#include "MCHRawElecMap/Mapper.h"

#include <iostream>

namespace o2::header
{
extern std::ostream& operator<<(std::ostream&, const o2::header::RAWDataHeaderV4&);
}

namespace o2::mch::raw
{

namespace impl
{
uint16_t CRUID_MASK = 0xFF;
uint16_t CHARGESUM_MASK = 0x100;

template <typename FORMAT, typename CHARGESUM>
struct PayloadDecoderImpl {

  using type = struct {
    void process(uint32_t, gsl::span<const std::byte>);
  };

  type operator()(const FeeLinkId& feeLinkId, SampaChannelHandler sampaChannelHandler, FeeLink2SolarMapper fee2solar);
};

template <typename CHARGESUM>
struct PayloadDecoderImpl<UserLogicFormat, CHARGESUM> {
  using type = UserLogicEndpointDecoder<CHARGESUM>;

  type operator()(const FeeLinkId& feeLinkId, SampaChannelHandler sampaChannelHandler, FeeLink2SolarMapper fee2solar)
  {
    return std::move(UserLogicEndpointDecoder<CHARGESUM>(feeLinkId.feeId(), fee2solar, sampaChannelHandler));
  }
};

template <typename CHARGESUM>
struct PayloadDecoderImpl<BareFormat, CHARGESUM> {
  using type = BareGBTDecoder<CHARGESUM>;

  type operator()(const FeeLinkId& feeLinkId, SampaChannelHandler sampaChannelHandler, FeeLink2SolarMapper fee2solar)
  {
    auto solarId = fee2solar(feeLinkId);
    if (!solarId.has_value()) {
      throw std::logic_error(fmt::format("{} could not get solarId from feelinkid={}\n", __PRETTY_FUNCTION__, feeLinkId));
    }
    return std::move(BareGBTDecoder<CHARGESUM>(solarId.value(), sampaChannelHandler));
  }
};

template <typename RDH>
void print(const RDH& rdh);

template <typename RDH, typename FORMAT, typename CHARGESUM>
class PageDecoderImpl
{
 public:
  PageDecoderImpl(SampaChannelHandler sampaChannelHandler, FeeLink2SolarMapper fee2solar) : mSampaChannelHandler{sampaChannelHandler},
                                                                                            mFee2SolarMapper(fee2solar)
  {
  }

  void operator()(Page page)
  {
    auto rdh = createRDH<RDH>(page);
    FeeLinkId feeLinkId(rdhFeeId(rdh) & CRUID_MASK, rdhLinkId(rdh));

    auto p = mPayloadDecoders.find(feeLinkId);

    if (p == mPayloadDecoders.end()) {
      mPayloadDecoders.emplace(feeLinkId, PayloadDecoderImpl<FORMAT, CHARGESUM>()(feeLinkId, mSampaChannelHandler, mFee2SolarMapper));
      p = mPayloadDecoders.find(feeLinkId);
    }

    uint32_t orbit = rdhOrbit(rdh);
    p->second.process(orbit, page.subspan(sizeof(rdh), rdhPayloadSize(rdh)));
  }

 private:
  SampaChannelHandler mSampaChannelHandler;
  FeeLink2SolarMapper mFee2SolarMapper;
  std::map<FeeLinkId, typename PayloadDecoderImpl<FORMAT, CHARGESUM>::type> mPayloadDecoders;
};

template <typename RDH>
class PageParser
{
 public:
  void operator()(RawBuffer buffer, PageDecoder pageDecoder)
  {
    size_t pos{0};
    while (pos < buffer.size_bytes() - sizeof(RDH)) {
      auto rdh = createRDH<RDH>(buffer.subspan(pos, sizeof(RDH)));
      auto payloadSize = rdhPayloadSize(rdh);
      pageDecoder(buffer.subspan(pos, sizeof(RDH) + payloadSize));
      pos += rdhOffsetToNext(rdh);
    }
  }
};

} // namespace impl
using V4 = o2::header::RAWDataHeaderV4;
//using V5 = o2::header::RAWDataHeaderV5;

template <>
void impl::print(const V4& rdh)
{
  std::cout << rdhOrbit(rdh) << " " << rdhBunchCrossing(rdh) << " " << rdhFeeId(rdh) << "\n";
}

PageDecoder createPageDecoder(RawBuffer rdhBuffer, SampaChannelHandler channelHandler, FeeLink2SolarMapper fee2solar)
{
  auto rdh = createRDH<V4>(rdhBuffer);
  if (isValid(rdh)) {
    if (rdhLinkId(rdh) == 15) {
      if (rdhFeeId(rdh) & impl::CHARGESUM_MASK) {
        return impl::PageDecoderImpl<V4, UserLogicFormat, ChargeSumMode>(channelHandler, fee2solar);
      } else {
        return impl::PageDecoderImpl<V4, UserLogicFormat, SampleMode>(channelHandler, fee2solar);
      }
    } else {
      if (rdhFeeId(rdh) & impl::CHARGESUM_MASK) {
        return impl::PageDecoderImpl<V4, BareFormat, ChargeSumMode>(channelHandler, fee2solar);
      } else {
        return impl::PageDecoderImpl<V4, BareFormat, SampleMode>(channelHandler, fee2solar);
      }
    }
  }
  throw std::invalid_argument("do not know how to create a page decoder for this RDH type\n");
}

PageDecoder createPageDecoder(RawBuffer rdhBuffer, SampaChannelHandler channelHandler)
{
  auto fee2solar = createFeeLink2SolarMapper<ElectronicMapperGenerated>();
  return createPageDecoder(rdhBuffer, channelHandler, fee2solar);
}

PageParser createPageParser(RawBuffer buffer)
{
  auto rdh = createRDH<V4>(buffer);
  if (isValid(rdh)) {
    return impl::PageParser<V4>();
  }
  throw std::invalid_argument("do not know how to create a page parser for this RDH type\n");
}

} // namespace o2::mch::raw
