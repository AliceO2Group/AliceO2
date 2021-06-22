// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_DECODED_DATA_HANDLES_H
#define O2_MCH_DECODED_DATA_HANDLES_H

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
/// a readout dual sampa channel and a SampaCluster containing the channel data.
using SampaChannelHandler = std::function<void(DsElecId dsId,
                                               DualSampaChannelId channel,
                                               SampaCluster)>;

/// A SampaHeartBeatHandler is a function that takes a chip index and
/// a bunch crossing counter value found in a HeartBeat packet
using SampaHeartBeatHandler = std::function<void(DsElecId dsId,
                                                 uint8_t chip,
                                                 uint20_t bunchCrossing)>;

/// A SampaErrorHandler is a function that takes a chip index and
/// a numerical code describing the error encountered during the decoding
using SampaErrorHandler = std::function<void(DsElecId dsId,
                                             int8_t chip,
                                             uint32_t error)>;

struct DecodedDataHandlers {
  SampaChannelHandler sampaChannelHandler;
  SampaHeartBeatHandler sampaHeartBeatHandler;
  SampaErrorHandler sampaErrorHandler;
};

} // namespace raw
} // namespace mch
} // namespace o2

#endif
