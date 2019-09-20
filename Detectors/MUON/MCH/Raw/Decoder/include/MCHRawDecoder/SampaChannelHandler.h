// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_SAMPA_CHANNEL_HANDLER_H
#define O2_MCH_RAW_SAMPA_CHANNEL_HANDLER_H

#include <functional>

#include "MCHRawCommon/SampaCluster.h"
#include "MCHRawElecMap/DsElecId.h"

namespace o2
{
namespace mch
{
namespace raw
{
/// A SampaChannelHandler is a function that takes a pair to identify
/// a readout sampa channel and a SampaCluster containing the channel data.
using SampaChannelHandler = std::function<void(DsElecId dsId,
                                               uint8_t channel,
                                               SampaCluster)>;
} // namespace raw
} // namespace mch
} // namespace o2

#endif
