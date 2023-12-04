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

#ifndef ALICEO2_ZDC_CONSTANTS_H
#define ALICEO2_ZDC_CONSTANTS_H

#include "CommonConstants/PhysicsConstants.h"
#include "CommonConstants/LHCConstants.h"
#include "CommonConstants/ZDCConstants.h"
#include <cstdint>
#include <cstdlib>
#include <array>
#include <string_view>
#include <string>
#include <type_traits>
#include <limits>

// Enable debug printout in reconstruction
// #define O2_ZDC_DEBUG
// TDC arrays in debug output
// #define O2_ZDC_TDC_C_ARRAY

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

constexpr int NTimeBinsPerBC = 12; //< number of samples per BC
constexpr int NBCReadOut = 4;      // N BCs read out per trigger
constexpr int NTimeBinsReadout = NTimeBinsPerBC * NBCReadOut;
constexpr int16_t Int16MaxVal = 0x7fff;

constexpr int NChannelsZN = 6;  //< number of channels stored per ZN
constexpr int NChannelsZP = 6;  //< number of channels stored per ZP
constexpr int NChannelsZEM = 2; //< number of channels stored per ZEM

constexpr float ChannelTimeBinNS = 2.; //< bin length in NS
constexpr float SampleLenghtNS = NTimeBinsPerBC * ChannelTimeBinNS;

constexpr int NChannels = 2 * (NChannelsZN + NChannelsZP) + NChannelsZEM;
constexpr uint8_t ALICETriggerMask = 0x1;

constexpr int NModules = 8;
constexpr int NChPerModule = 4;
constexpr int NLinks = NModules * 2;
constexpr int NDigiChannels = NModules * NChPerModule;
constexpr int NWPerBc = 3;
constexpr int MaxTriggerChannels = NChannels;

// Limits
constexpr int ADCMin = -2048, ADCMax = 2047, ADCRange = 4096; // 12 bit ADC
constexpr float FInfty = std::numeric_limits<float>::infinity();
constexpr float DInfty = std::numeric_limits<double>::infinity();

// Encoding of ZDC energy into an uint32_t value
// Most significant 5 bits are for channel id, least significant 27 bits are for energy
// with offset and unit as specified below
constexpr float EnergyOffset = 10000; // Energy offset (GeV)
constexpr float EnergyUnit = 0.01;    // Energy unit (GeV)
constexpr uint32_t EnergyMask = 0x07ffffff;
constexpr uint32_t EnergyChMask = 0xf8000000;

// Temporary reconstructed event
constexpr int MaxTDCValues = 5;  // max number of TDC values to store in reconstructed event
constexpr int NTDCChannels = 10; // max number of TDC values to store in reconstructed event
constexpr uint32_t ZDCRefInitVal = 0xffffffff;
// Parameters of interpolating function
constexpr int TSL = 6;                    // number of zeros on the right (and on the left) of central peak
constexpr int TSN = 200;                  // Number of interpolated points between each pair = TSN-1
constexpr int TSNH = TSN / 2;             // Half of TSN
constexpr int NTS = 2 * TSL * TSN + 1;    // Tapered sinc function array size
constexpr float FTDCAmp = 1. / 8.;        // Multiplication factor in conversion from integer - TODO increase precision assuming Amplitude>0
constexpr int NIS = NTimeBinsPerBC * TSN; // Number of interpolated samples
// With a reference clock of 40 MHz exact this FTDCVal would have been
// constexpr float FTDCVal = 1. / TSNS;
// with constexpr int TSNS = 96;
// However we need to modify to take into account actual LHC clock frequency
// Multiplication factor in conversion from integer
constexpr float FTDCVal = o2::constants::lhc::LHCBunchSpacingNS / NTimeBinsPerBC / TSN;
constexpr float FOffset = 8.; // Conversion from average pedestal to representation in OrbitData (16 bit)

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

constexpr int NBucket = 10; // Number of buckets in a
constexpr int NBKZero = 5;  // Bucket with main-main collisions
constexpr int NFParA = 3;   // Number of parameters in fitting function - Amplitude
constexpr int NFParT = 3;   // Number of parameters in fitting function - Time
constexpr int NBCAn = 3;    // Number of analyzed bunches

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

constexpr uint32_t MaskZNA = 0x0000001f;
constexpr uint32_t MaskAllZNA = 0x0000003f;
constexpr uint32_t MaskZPA = 0x000007c0;
constexpr uint32_t MaskAllZPA = 0x00000fc0;
constexpr uint32_t MaskZEM = 0x00003000;
constexpr uint32_t MaskZNC = 0x000fc000;
constexpr uint32_t MaskAllZNC = 0x0007f000;
constexpr uint32_t MaskZPC = 0x01f00000;
constexpr uint32_t MaskAllZPC = 0x03f00000;

constexpr std::string_view ChannelNames[] = {
  "ZNAC", //  0
  "ZNA1", //  1
  "ZNA2", //  2
  "ZNA3", //  3
  "ZNA4", //  4
  "ZNAS", //  5
  //
  "ZPAC", //  6
  "ZPA1", //  7
  "ZPA2", //  8
  "ZPA3", //  9
  "ZPA4", // 10
  "ZPAS", // 11
  //
  "ZEM1", // 12
  "ZEM2", // 13
  //
  "ZNCC", // 14
  "ZNC1", // 15
  "ZNC2", // 16
  "ZNC3", // 17
  "ZNC4", // 18
  "ZNCS", // 19
  //
  "ZPCC", // 20
  "ZPC1", // 21
  "ZPC2", // 22
  "ZPC3", // 23
  "ZPC4", // 24
  "ZPCS"  // 25
};

// From TDC ID to signal ID
const int TDCSignal[NTDCChannels] = {
  IdZNAC,   // TDCZNAC
  IdZNASum, // TDCZNAS
  IdZPAC,   // TDCZPAC
  IdZPASum, // TDCZPAS
  IdZEM1,   // TDCZEM1
  IdZEM2,   // TDCZEM2
  IdZNCC,   // TDCZNCC
  IdZNCSum, // TDCZNCS
  IdZPCC,   // TDCZPCC
  IdZPCSum  // TDCZPCS
};

// From Signal ID to TDC ID
const int SignalTDC[NChannels] = {
  TDCZNAC,
  TDCZNAS, TDCZNAS, TDCZNAS, TDCZNAS, TDCZNAS,
  TDCZPAC,
  TDCZPAS, TDCZPAS, TDCZPAS, TDCZPAS, TDCZPAS,
  TDCZEM1,
  TDCZEM2,
  TDCZNCC,
  TDCZNCS, TDCZNCS, TDCZNCS, TDCZNCS, TDCZNCS,
  TDCZPCC,
  TDCZPCS, TDCZPCS, TDCZPCS, TDCZPCS, TDCZPCS};

constexpr int DbgZero = 0;
constexpr int DbgMinimal = 1;
constexpr int DbgMedium = 2;
constexpr int DbgFull = 3;
constexpr int DbgExtra = 4;

// paths to CCDB objects
// TODO: eventually these paths should be retrieved from NameConfigurator class
// TODO: we would better use "constexpr string_view" here, but it makes sense only if
// TODO: CcdbApi and BasicCCDBManager would also take string_view as an argument

const std::string CCDBPathConfigSim = "ZDC/Config/Sim";
const std::string CCDBPathConfigModule = "ZDC/Config/Module";
const std::string CCDBPathRecoConfigZDC = "ZDC/Calib/RecoConfigZDC";
const std::string CCDBPathTDCCalib = "ZDC/Calib/TDCCalib";
const std::string CCDBPathTDCCalibConfig = "ZDC/Calib/TDCCalibConfig";
const std::string CCDBPathTDCCorr = "ZDC/Calib/TDCCorr";
const std::string CCDBPathEnergyCalib = "ZDC/Calib/EnergyCalib";
const std::string CCDBPathTowerCalib = "ZDC/Calib/TowerCalib";
const std::string CCDBPathInterCalibConfig = "ZDC/Calib/InterCalibConfig";
const std::string CCDBPathWaveformCalib = "ZDC/Calib/WaveformCalib";
const std::string CCDBPathWaveformCalibConfig = "ZDC/Calib/WaveformCalibConfig";
const std::string CCDBPathBaselineCalib = "ZDC/Calib/BaselineCalib";
const std::string CCDBPathBaselineCalibConfig = "ZDC/Calib/BaselineCalibConfig";
const std::string CCDBPathNoiseCalib = "ZDC/Calib/NoiseCalib";

enum Ped { PedND = 0,
           PedEv = 1,
           PedOr = 2,
           PedQC = 3,
           PedMissing = 4 };

// Max 256 error messages
enum Msg { MsgGeneric = 0,
           MsgTDCPedQC = 1,
           MsgTDCPedMissing = 2,
           MsgADCPedOr = 3,
           MsgADCPedQC = 4,
           MsgADCPedMissing = 5,
           MsgOffPed = 6,
           MsgPilePed = 7,
           MsgPileTM = 8,
           MsgADCMissingwTDC = 9,
           MsgTDCPileEvC = 10, // A correction is done
           MsgTDCPileEvE = 11, // Correction has problems
           MsgTDCPileM1C = 12,
           MsgTDCPileM1E = 13,
           MsgTDCPileM2C = 14,
           MsgTDCPileM2E = 15,
           MsgTDCPileM3C = 16,
           MsgTDCPileM3E = 17,
           MsgTDCSigE = 18, // Error correcting isolated signal
           MsgEnd           // End_of_messages
};

constexpr std::string_view MsgText[] = {
  "Generic E",
  "TDC QC ped",
  "TDC missing ped",
  "ADC Orbit ped",
  "ADC QC ped",
  "ADC missing ped",
  "Positive ped offset",
  "Pile-up in ev ped",
  "Pile-up in TM",
  "ADC missing, TDC present",
  "TDC pile-up Ev C", // In-event pile-up corrected
  "TDC pile-up Ev E", // In-event pile-up correction error
  "TDC pile-up M1 C", // Corrected for pile-up in bunch -1
  "TDC pile-up M1 E",
  "TDC pile-up M2 C",
  "TDC pile-up M2 E",
  "TDC pile-up M3 C",
  "TDC pile-up M3 E",
  "TDC signal E"
  // End_of_messages
};

// List of channels that can be calibrated
constexpr std::array<int, 10> ChEnergyCalib{IdZNAC, IdZNASum, IdZPAC, IdZPASum,
                                            IdZEM1, IdZEM2,
                                            IdZNCC, IdZNCSum, IdZPCC, IdZPCSum};

constexpr std::array<int, 17> ChTowerCalib{IdZNA1, IdZNA2, IdZNA3, IdZNA4,
                                           IdZPA1, IdZPA2, IdZPA3, IdZPA4,
                                           IdZNC1, IdZNC2, IdZNC3, IdZNC4,
                                           IdZPC1, IdZPC2, IdZPC3, IdZPC4,
                                           IdZEM2};

constexpr std::array<int, NChannels> CaloCommonPM{IdZNAC, IdZNAC, IdZNAC, IdZNAC, IdZNAC, IdZNAC,
                                                  IdZPAC, IdZPAC, IdZPAC, IdZPAC, IdZPAC, IdZPAC,
                                                  IdZEM1, IdZEM2,
                                                  IdZNCC, IdZNCC, IdZNCC, IdZNCC, IdZNCC, IdZNCC,
                                                  IdZPCC, IdZPCC, IdZPCC, IdZPCC, IdZPCC, IdZPCC};

// Placeholders
constexpr int DummyIntRange = -NTimeBinsPerBC - 1;

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

// Calibration workflows
// Waveform calibration
constexpr int WaveformCalib_NBB = 3; // Number of bunches acquired before colliding b.c.
constexpr int WaveformCalib_NBA = 6; // Number of bunches acquired after colliding b.c.
constexpr int WaveformCalib_NBT = WaveformCalib_NBB + WaveformCalib_NBA + 1;
constexpr int WaveformCalib_NW = WaveformCalib_NBT * NIS;

using zdcBaseline_t = int16_t;
constexpr int BaselineMin = -32768, BaselineMax = 32767, BaselineRange = 65536; // 16 bit with sign

} // namespace zdc
} // namespace o2

#endif
