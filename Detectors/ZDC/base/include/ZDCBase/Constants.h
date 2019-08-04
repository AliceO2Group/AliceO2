// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ZDC_CONSTANTS_H
#define ALICEO2_ZDC_CONSTANTS_H

#include "CommonConstants/PhysicsConstants.h"

namespace o2
{
namespace zdc
{

enum DetectorID { DetIDOffs = 1,
                  ZNA = 1,
                  ZPA = 2,
                  ZEM = 3,
                  ZNC = 4,
                  ZPC = 5 }; // IDs of subdetector
enum ChannelTypeZNP { Common,
                      Ch1,
                      Ch2,
                      Ch3,
                      Ch4,
                      Sum }; // channel IDs for ZN and ZP
enum ChannelTypeZEM { ZEMCh1,
                      ZEMCh2 }; //  channel IDs for ZEMs

constexpr int NTimeBinsPerBC = 12;                   //< number of samples per BC
constexpr int NBCBefore = 1;                         //< number of BCs read before the triggered BC
constexpr int NBCAfter = 2;                          //< number of BCs read after the triggered BC
constexpr int NBCReadOut = 1 + NBCBefore + NBCAfter; // N BCs read out per trigger

constexpr int NChannelsZN = 6;  //< number of channels stored per ZN
constexpr int NChannelsZP = 6;  //< number of channels stored per ZP
constexpr int NChannelsZEM = 2; //< number of channels stored per ZEM

constexpr float ChannelTimeBinNS = 2.; //< bin length in NS
constexpr float SampleLenghtNS = NTimeBinsPerBC * ChannelTimeBinNS;

constexpr int NChannels = 2 * (NChannelsZN + NChannelsZP) + NChannelsZEM;

//< get detector TOF correction in ns
constexpr float getTOFCorrection(int det)
{
  constexpr float TOFCorr[5] = {
    11253.3 / o2::constants::physics::LightSpeedCm2NS,
    11251.8 / o2::constants::physics::LightSpeedCm2NS,
    760. / o2::constants::physics::LightSpeedCm2NS,
    11261.3 / o2::constants::physics::LightSpeedCm2NS,
    11253.3 / o2::constants::physics::LightSpeedCm2NS};
  return TOFCorr[det - DetIDOffs];
}

//< map detector/tower to continuous channel ID
constexpr int toChannel(int det, int tower)
{
  constexpr int DetChMap[5][6] = {{0, 1, 2, 3, 4, 5},        // ZNA
                                  {6, 7, 8, 9, 10, 11},      // ZPA
                                  {12, 13, -1, -1, -1, -1},  // ZEM
                                  {14, 15, 16, 17, 18, 19},  // ZNC
                                  {20, 21, 22, 23, 24, 25}}; // ZPC
  return DetChMap[det - 1][tower];
}

//< map channelID to detector/tower
constexpr int toDet(int channel, int& tower)
{
  if (channel < toChannel(ZNC, 0)) {
    tower = channel % NChannelsZN;
    return DetIDOffs + channel / NChannelsZP;
  } else {
    channel -= toChannel(ZNC, 0);
    tower = channel % NChannelsZN;
    return ZNC + channel / NChannelsZP;
  }
}

} // namespace zdc
} // namespace o2

#endif
