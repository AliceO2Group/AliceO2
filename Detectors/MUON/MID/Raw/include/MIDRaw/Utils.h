// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/Utils.h
/// \brief  Raw data utilities for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   19 November 2019
#ifndef O2_MID_UTILS_H
#define O2_MID_UTILS_H

#include "Headers/RDHAny.h"
#include "DetectorsRaw/RDHUtils.h"

namespace o2
{
namespace mid
{
namespace raw
{
static constexpr uint8_t sUserLogicLinkID = 15; // Link ID for the user logic

/// Test if the data comes from the common logic
inline bool isBare(const o2::header::RDHAny& rdh) { return (o2::raw::RDHUtils::getLinkID(rdh) != sUserLogicLinkID); }

} // namespace raw
} // namespace mid
} // namespace o2

#endif /* O2_MID_UTILS_H */
