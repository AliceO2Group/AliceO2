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
#include <cstdint>
#include <string_view>
#include <string>
#include <type_traits>

namespace o2
{
namespace zdc
{

enum DetectorID { DetIDOffs = 1,
                  ZNA = 1,
                  ZPA = 2,
                  ZEM = 3,
                  ZNC = 4,
                  ZPC = 5,
                  MinDetID = DetIDOffs,
                  MaxDetID = 5 }; // IDs of subdetector
enum ChannelTypeZNP { Common,
                      Ch1,
                      Ch2,
                      Ch3,
                      Ch4,
                      Sum }; // channel IDs for ZN and ZP
enum ChannelTypeZEM { ZEMCh1,
                      ZEMCh2 }; //  channel IDs for ZEMs

constexpr int NTimeBinsPerBC = 12;                   //< number of samples per BC
constexpr int NBCReadOut = 4;                        // N BCs read out per trigger
constexpr int NTimeBinsReadout = NTimeBinsPerBC * NBCReadOut;

constexpr int NChannelsZN = 6;  //< number of channels stored per ZN
constexpr int NChannelsZP = 6;  //< number of channels stored per ZP
constexpr int NChannelsZEM = 2; //< number of channels stored per ZEM

constexpr float ChannelTimeBinNS = 2.; //< bin length in NS
constexpr float SampleLenghtNS = NTimeBinsPerBC * ChannelTimeBinNS;

constexpr int NChannels = 2 * (NChannelsZN + NChannelsZP) + NChannelsZEM;
constexpr uint32_t AllChannelsMask = (0x1 << NChannels) - 1;

constexpr int NModules = 8;
constexpr int MaxTriggerChannels = 10;

constexpr int MaxTDCValues = 5;  // max number of TDC values to store in reconstructed event
constexpr int NTDCChannels = 10; // max number of TDC values to store in reconstructed event
enum TDCChannelID {
  TDCZNAC,
  TDCZNAS,
  TDCZPAC,
  TDCZPAS,
  TDCZEM1,
  TDCZEM2,
  TDCZNCC,
  TDCZNCS,
  TDCZPCC,
  TDCZPCS
}; // TDC channels in reconstructed event, their number should be equal to NTDCChannels

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

//< map detector/tower to continuous channel Id
constexpr int IdDummy = -1;
constexpr int IdVoid = -2;

constexpr int IdZNAC = 0;
constexpr int IdZNA1 = 1;
constexpr int IdZNA2 = 2;
constexpr int IdZNA3 = 3;
constexpr int IdZNA4 = 4;
constexpr int IdZNASum = 5;
//
constexpr int IdZPAC = 6;
constexpr int IdZPA1 = 7;
constexpr int IdZPA2 = 8;
constexpr int IdZPA3 = 9;
constexpr int IdZPA4 = 10;
constexpr int IdZPASum = 11;
//
constexpr int IdZEM1 = 12;
constexpr int IdZEM2 = 13;
//
constexpr int IdZNCC = 14;
constexpr int IdZNC1 = 15;
constexpr int IdZNC2 = 16;
constexpr int IdZNC3 = 17;
constexpr int IdZNC4 = 18;
constexpr int IdZNCSum = 19;
//
constexpr int IdZPCC = 20;
constexpr int IdZPC1 = 21;
constexpr int IdZPC2 = 22;
constexpr int IdZPC3 = 23;
constexpr int IdZPC4 = 24;
constexpr int IdZPCSum = 25;

constexpr std::string_view ChannelNames[] = {
  "ZNAC",
  "ZNA1",
  "ZNA2",
  "ZNA3",
  "ZNA4",
  "ZNAS",
  //
  "ZPAC",
  "ZPA1",
  "ZPA2",
  "ZPA3",
  "ZPA4",
  "ZPAS",
  //
  "ZEM1",
  "ZEM2",
  //
  "ZNCC",
  "ZNC1",
  "ZNC2",
  "ZNC3",
  "ZNC4",
  "ZNCS",
  //
  "ZPCC",
  "ZPC1",
  "ZPC2",
  "ZPC3",
  "ZPC4",
  "ZPCS"};

// paths to CCDB objects
// TODO: eventually these paths should be retrieved from NameConfigurator class
// TODO: we would better use "constexpr string_view" here, but it makes sense only if
// TODO: CcdbApi and BasicCCDBManager would also take string_view as an argument

const std::string CCDBPathConfigSim = "ZDC/Config/Sim";
const std::string CCDBPathConfigModule = "ZDC/Config/Module";

constexpr std::string_view DummyName = "Dumm";
constexpr std::string_view VoidName = " NA ";

constexpr int toChannel(int det, int tower)
{
  constexpr int DetChMap[5][6] = {{IdZNAC, IdZNA1, IdZNA2, IdZNA3, IdZNA4, IdZNASum},   // ZNA
                                  {IdZPAC, IdZPA1, IdZPA2, IdZPA3, IdZPA4, IdZPASum},   // ZPA
                                  {IdDummy, IdZEM1, IdZEM2, IdDummy, IdDummy, IdDummy}, // ZEM
                                  {IdZNCC, IdZNC1, IdZNC2, IdZNC3, IdZNC4, IdZNCSum},   // ZNC
                                  {IdZPCC, IdZPC1, IdZPC2, IdZPC3, IdZPC4, IdZPCSum}};  // ZPC
  return DetChMap[det - 1][tower];
}

constexpr const char* channelName(int channel)
{

  // sanity check
  static_assert(NChannels == sizeof(ChannelNames) / sizeof(std::string_view), "Channels definition is not complete");

  if (channel >= 0 && channel < NChannels) {
    return ChannelNames[channel].data();
  } else if (channel == IdDummy) {
    return DummyName.data();
  }
  return VoidName.data();
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
