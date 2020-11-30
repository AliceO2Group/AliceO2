// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_DECODER_PAGE_DECODER_H
#define O2_MCH_RAW_DECODER_PAGE_DECODER_H

#include <functional>
#include <gsl/span>
#include <map>
#include "MCHRawDecoder/DecodedDataHandlers.h"
#include "MCHRawElecMap/Mapper.h"

namespace o2::mch::raw
{

// A (CRU) Page is a raw memory buffer containing a pair (RDH,payload)
using Page = gsl::span<const std::byte>;

// A PageDecoder decodes a single (CRU) Page
using PageDecoder = std::function<void(Page buffer)>;

using RawBuffer = gsl::span<const std::byte>;

// Create a PageDecoder depending on the first rdh found in the buffer.
//
// @param rdhBuffer a raw memory buffer containing (at least) one RDH
// which information is used to decide which PageDecoder implementation to choose
// @param decodedDataHandlers a structure with various callable objects (optional) that
/// will be called for each decoded Sampa packet and in case of decoding errors
//
PageDecoder createPageDecoder(RawBuffer rdhBuffer,
                              DecodedDataHandlers decodedDataHandlers);

// Same as above but only to be used for special cases, e.g. when
// trying to decode test beam data with an electronic mapping that
// does not match the expected one for Run3.
//
// @param fee2solar (optional) a callable object that will convert a FeeLinkId
// object into a solarId.
//
PageDecoder createPageDecoder(RawBuffer rdhBuffer,
                              DecodedDataHandlers decodedDataHandlers,
                              FeeLink2SolarMapper fee2solar);

// Alternative versions of the same functions, taking a SampaChannelHandler as parameter.
[[deprecated("Use createPageDecoder(RawBuffer,DecodedDataHandlers) instead.")]] PageDecoder createPageDecoder(RawBuffer rdhBuffer,
                                                                                                              SampaChannelHandler channelHandler);

[[deprecated("Use createPageDecoder(RawBuffer,DecodedDataHandlers,fee2solar) instead.")]] PageDecoder createPageDecoder(RawBuffer rdhBuffer,
                                                                                                                        SampaChannelHandler channelHandler,
                                                                                                                        FeeLink2SolarMapper fee2solar);

// A PageParser loops over the given buffer and apply the given page decoder
// to each page.
using PageParser = std::function<void(RawBuffer buffer, PageDecoder pageDecoder)>;

// Create a PageParser
PageParser createPageParser();

} // namespace o2::mch::raw

#endif
