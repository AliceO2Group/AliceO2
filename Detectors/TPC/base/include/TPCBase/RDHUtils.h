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

#ifndef AliceO2_TPC_RDHUtils_H
#define AliceO2_TPC_RDHUtils_H

#include "DetectorsRaw/RDHUtils.h"
//#include "Headers/RAWDataHeader.h"

namespace o2
{
namespace tpc
{
namespace rdh_utils
{

using o2::raw::RDHUtils;
using FEEIDType = uint16_t;
static constexpr FEEIDType UserLogicLinkID = 15; ///< virtual link ID for ZS data
static constexpr FEEIDType IDCLinkID = 20;       ///< Identifier for integrated digital currents
static constexpr FEEIDType ILBZSLinkID = 21;     ///< Identifier for improved link-based ZS
static constexpr FEEIDType SACLinkID = 25;       ///< Identifier for sampled analog currents

/// compose feeid from cru, endpoint and link
static constexpr FEEIDType getFEEID(const FEEIDType cru, const FEEIDType endpoint, const FEEIDType link) { return FEEIDType((cru << 7) | ((endpoint & 1) << 6) | (link & 0x3F)); }
template <typename T>
static constexpr FEEIDType getFEEID(const T cru, const T endpoint, const T link)
{
  return getFEEID(FEEIDType(cru), FEEIDType(endpoint), FEEIDType(link));
}

/// extract cru number from feeid
static constexpr FEEIDType getCRU(const FEEIDType feeID) { return (feeID >> 7); }

/// extract endpoint from feeid
static constexpr FEEIDType getEndPoint(const FEEIDType feeID) { return (feeID >> 6) & 0x1; }

/// extract endpoint from feeid
static constexpr FEEIDType getLink(const FEEIDType feeID) { return feeID & 0x3F; }

/// extract cru, endpoint and link from feeid
static constexpr void getMapping(const FEEIDType feeID, FEEIDType& cru, FEEIDType& endpoint, FEEIDType& link)
{
  cru = getCRU(feeID);
  endpoint = getEndPoint(feeID);
  link = getLink(feeID);
}

/// if link in feeID is from user logic
static constexpr bool isFromUserLogic(const FEEIDType feeID) { return (getLink(feeID) == UserLogicLinkID); }

/// extract cru number from RDH
template <typename RDH>
static constexpr FEEIDType getCRU(const RDH& rdh)
{
  return getCRU(RDHUtils::getFEEID(rdh));
}

/// extract endpoint from RDH
template <typename RDH>
static constexpr FEEIDType getEndPoint(const RDH& rdh)
{
  return getEndPoint(RDHUtils::getFEEID(rdh));
}

/// extract link from RDH
template <typename RDH>
static constexpr FEEIDType getLink(const RDH& rdh)
{
  return getLink(RDHUtils::getFEEID(rdh));
}

/// if link in feeID is from user logic
template <typename RDH>
static constexpr bool isFromUserLogic(const RDH& rdh)
{
  return isFromUserLogic(RDHUtils::getFEEID(rdh));
}

template <typename RDH, typename T>
static constexpr void setFEEID(RDH& rdh, const T cru, const T endpoint, const T link)
{
  RDHUtils::setFEEID(rdh, getFEEID(cru, endpoint, link));
}

/// extract cru, endpoint and link from RDH
template <typename RDH>
static constexpr void getMapping(const RDH& rdh, FEEIDType& cru, FEEIDType& endpoint, FEEIDType& link)
{
  cru = getCRU(rdh);
  endpoint = getEndPoint(rdh);
  link = getLink(rdh);
}

} // namespace rdh_utils
} // namespace tpc
} // namespace o2

#endif
