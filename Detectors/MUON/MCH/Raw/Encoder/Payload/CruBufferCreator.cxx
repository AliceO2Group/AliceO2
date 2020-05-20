// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CruBufferCreator.h"

namespace o2::mch::raw::test
{

std::vector<std::byte> fillChargeSum(PayloadEncoder& encoder, int norbit,
                                     uint32_t firstOrbit, uint16_t firstBC)
{
  uint16_t sampaTime{24};
  uint32_t bunchCrossing = 567;
  uint16_t bc(firstBC);

  encoder.startHeartbeatFrame(firstOrbit, bc);

  encoder.addChannelData(DsElecId{728, 1, 0}, 3, {SampaCluster(sampaTime, bunchCrossing, 13)});
  encoder.addChannelData(DsElecId{728, 1, 0}, 13, {SampaCluster(sampaTime, bunchCrossing, 133)});
  encoder.addChannelData(DsElecId{728, 1, 0}, 23, {SampaCluster(sampaTime, bunchCrossing, 163)});

  encoder.addChannelData(DsElecId{361, 0, 4}, 0, {SampaCluster(sampaTime, bunchCrossing, 10)});
  encoder.addChannelData(DsElecId{361, 0, 4}, 1, {SampaCluster(sampaTime, bunchCrossing, 20)});
  encoder.addChannelData(DsElecId{361, 0, 4}, 2, {SampaCluster(sampaTime, bunchCrossing, 30)});
  encoder.addChannelData(DsElecId{361, 0, 4}, 3, {SampaCluster(sampaTime, bunchCrossing, 40)});

  encoder.addChannelData(DsElecId{448, 6, 2}, 22, {SampaCluster(sampaTime, bunchCrossing, 420)});
  encoder.addChannelData(DsElecId{448, 6, 2}, 23, {SampaCluster(sampaTime, bunchCrossing, 430)});
  encoder.addChannelData(DsElecId{448, 6, 2}, 24, {SampaCluster(sampaTime, bunchCrossing, 440)});
  encoder.addChannelData(DsElecId{448, 6, 2}, 25, {SampaCluster(sampaTime, bunchCrossing, 450)});
  encoder.addChannelData(DsElecId{448, 6, 2}, 26, {SampaCluster(sampaTime, bunchCrossing, 460)});
  encoder.addChannelData(DsElecId{448, 6, 2}, 42, {SampaCluster(sampaTime, bunchCrossing, 420)});

  if (norbit > 1) {
    encoder.startHeartbeatFrame(firstOrbit + 1, bc);
    encoder.addChannelData(DsElecId{728, 1, 2}, 0, {SampaCluster(sampaTime, bunchCrossing, 10)});
    encoder.addChannelData(DsElecId{728, 1, 2}, 1, {SampaCluster(sampaTime, bunchCrossing, 10)});
    encoder.addChannelData(DsElecId{361, 0, 4}, 0, {SampaCluster(sampaTime, bunchCrossing, 10)});
    encoder.addChannelData(DsElecId{361, 0, 4}, 1, {SampaCluster(sampaTime, bunchCrossing, 20)});
    encoder.addChannelData(DsElecId{361, 0, 4}, 2, {SampaCluster(sampaTime, bunchCrossing, 30)});
    encoder.addChannelData(DsElecId{361, 0, 4}, 3, {SampaCluster(sampaTime, bunchCrossing, 40)});
  }

  if (norbit > 2) {
    encoder.startHeartbeatFrame(firstOrbit + 2, bc);
    encoder.addChannelData(DsElecId{448, 6, 2}, 12, {SampaCluster(sampaTime, bunchCrossing, 420)});
  }

  std::vector<std::byte> buffer;
  encoder.moveToBuffer(buffer);
  return buffer;
}

template <typename FORMAT>
struct CruBufferCreator<FORMAT, ChargeSumMode> {
  static std::vector<std::byte> makeBuffer(int norbit = 1,
                                           uint32_t firstOrbit = 12345, uint16_t firstBC = 678)
  {
    auto encoder = createPayloadEncoder<FORMAT, ChargeSumMode, true>();

    return fillChargeSum(*(encoder.get()), norbit, firstOrbit, firstBC);
  }
};

template struct CruBufferCreator<BareFormat, ChargeSumMode>;

template struct CruBufferCreator<UserLogicFormat, ChargeSumMode>;
} // namespace o2::mch::raw::test
