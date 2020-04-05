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
#include "MCHRawDecoder/SampaChannelHandler.h"

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
// @param channelHandler (optional) a callable object that will be called for each
// decoded SampaCluster
//
// Note that while the channelHandler parameter is optional, unless for testing purposes
// it does not really make sense to not provide it.
//
PageDecoder createPageDecoder(RawBuffer rdhBuffer,
                              SampaChannelHandler channelHandler = nullptr);

// A PageParser loops over the given buffer and apply the given page decoder
// to each page.
using PageParser = std::function<void(RawBuffer buffer, PageDecoder pageDecoder)>;

// Create a PageParser depending on the first rdh found in the buffer.
PageParser createPageParser(RawBuffer rdhBuffer);

} // namespace o2::mch::raw

#endif
