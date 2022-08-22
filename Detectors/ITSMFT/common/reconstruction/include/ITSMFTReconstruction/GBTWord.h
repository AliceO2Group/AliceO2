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

#ifndef ALICEO2_ITSMFT_GBTWORD_H
#define ALICEO2_ITSMFT_GBTWORD_H

// \file GBTWord.h
// \brief Classes for creation/interpretation of ITS/MFT GBT data

#include <Rtypes.h>
#include <cstdint>

namespace o2
{
namespace itsmft
{

constexpr uint64_t LANESMask = (0x1 << 28) - 1; // at most 28 lanes

/// GBT payload header flag
constexpr uint8_t GBTFlagDataHeader = 0xe0;
/// GBT payload trailer flag
constexpr uint8_t GBTFlagDataTrailer = 0xf0;
/// GBT trigger status word flag
constexpr uint8_t GBTFlagTrigger = 0xe8;
/// GBT diagnostic status word flag
constexpr uint8_t GBTFlagDiagnostic = 0xe4;
/// GBT calibration status word flag
constexpr uint8_t GBTFlagCalibration = 0xf8;

// GBT header flag in the RDH
constexpr uint8_t GBTFlagRDH = 0x00;
// GBT header flag for the ITS IB: 001 bbbbb with bbbbb -> Lane Number (0-8)
constexpr uint8_t GBTFlagDataIB = 0x20;
// GBT header flag for the ITS OB: 010 bb ccc with bb -> Connector Number (00,01,10,11), ccc -> Lane Number (0-6)
constexpr uint8_t GBTFlagDataOB = 0x40;
// GBT header flag for the MFT
constexpr uint8_t GBTFlagDataMFT = 0x80;
// GBT header flag for the ITS IB idagnostic : 101 bbbbb with bbbbb -> Lane Number (0-8)
constexpr uint8_t GBTFlagDiagnosticIB = 0xa0;
// GBT header flag for the ITS OB diagnostic word: 110 bb ccc with bb -> Connector Number (00,01,10,11), ccc -> Lane Number (0-6)
constexpr uint8_t GBTFlagDiagnosticOB = 0x60;

// GBT header flag for the ITS Status word : 111 bbbbb with bbbbb -> Lane Number
constexpr uint8_t GBTFlagStatus = 0xe0;

constexpr int GBTWordLength = 10;       // lentgh in bytes
constexpr int GBTPaddedWordLength = 16; // lentgh in bytes with padding

struct GBTWord {
  /// GBT word of 80 bits, bits 72:79 are reserver for GBT Header flag, the rest depends on specifications
  union {
    // RS: packing will be needed only if some of the members cross 64 bit boundary
    struct /*__attribute__((packed))*/ {
      uint64_t activeLanes : 28; /// 0:27   Bit map of lanes active and eligible for readout
      uint64_t na0hn : 36;       /// 28:71  reserved
      uint64_t na1hn : 8;        /// 28:71  reserved
      uint64_t id : 8;           /// 72:79  0xe0; Header Status Word (HSW) identifier
    };                           // HEADER NEW

    struct /*__attribute__((packed))*/ {
      uint64_t packetIdx : 16;    ///  0:15  Index of Data Packet within trigger
      uint64_t activeLanesL : 28; /// 16:43  Bit map of lanes active and eligible for readout
      uint64_t na0h : 20;         /// 44:64  reserved
      uint64_t na1h : 8;          /// 64:71  reserved
      //      uint64_t id : 8;           /// 72:79  0xe0; Header Status Word (HSW) identifier
    }; // HEADER Legacy

    struct /*__attribute__((packed))*/ {
      uint64_t lanesStops : 28;         ///  0:27  Bit map of “Valid Lane stops received”, 1 bit per lane, NOT USED
      uint64_t lanesTimeout : 28;       /// 28:55  Bit map of “Lane timeouts received”, 1 bit per lane, NOT USED
      uint64_t na1t : 8;                /// 56:63  reserved
      uint64_t packetDone : 1;          /// 64     = 1 when current trigger packets transmission done
      uint64_t transmissionTimeout : 1; /// 65     = 1 if timeout while waiting for data on lanes
      uint64_t packetOverflow : 1;      /// 66     = 1 if max number of packets reached
      uint64_t laneStartsViolation : 1; /// 67     = 1 if at least 1 lane (eligible for readout) had a “start violation”
      uint64_t laneTimeouts : 1;        /// 68     = 1 if at least 1 lane (eligible for readout) had a “start violation”
      uint64_t na2t : 3;                /// 69:71  reserved
      //  uint8_t  id : 8;                /// = 0xf0; Trailer Status Word (TSW) identifier
    }; // TRAILER

    struct /*__attribute__((packed))*/ {
      uint64_t triggerType : 12; /// 0:11   12 lowest bits of trigger type received from CTP
      uint64_t internal : 1;     /// 12     Used in Continuous Mode for internally generated trigger
      uint64_t noData : 1;       /// 13     No data expected (too close to previous trigger or error)
      uint64_t continuation : 1; /// 14     following data is continuation of the trigger from the previous CRU page
      uint64_t na1tr : 1;        /// 15     reserved
      uint64_t bc : 12;          /// 16:27  HB or internal trigger BC count or trigger BC from CTP
      uint64_t na2tr : 4;        /// 28:31  reserved
      uint64_t orbit : 32;       /// 32:63  Last received HB Orbit or trigger orbit count/ Orbit as received from CTP
      uint64_t na3tr : 6;        /// 64:71  reserved
      //  uint8_t  id : 8;                /// = 0xc0; Trigger Status Word (TSW) identifier
    }; // TRIGGER

    struct /*__attribute__((packed))*/ {
      uint64_t na0diag : 64; ///
      //      uint64_t id : 8;           /// 72:79  0xe4; diagnostic word identifier
    }; // HEADER Legacy

    struct __attribute__((packed)) {
      uint64_t calibUserField : 48; /// 0:47   user field
      uint64_t calibCounter : 24;   /// 48:71  elf-incrementing counter of
      //  uint64_t id : 8;            /// 72:79  0xf8; Calibration Status Word (HSW) identifier
    }; /// Calibration Data Word

    struct {
      uint64_t diagnosticData : 64; /// 0:63   Error specific diagnostic data
      uint64_t laneErrorID : 8;     /// 64:71  Identifier of the specific error condition
      //      uint64_t id : 8;           /// 72:79  0xe0;  Status Word (HSW) identifier
    }; // HEADER Legacy

    uint8_t data8[16]; // 80 bits GBT word + optional padding to 128 bits
    uint64_t data64[2] = {0};
  };

  GBTWord() = default;

  /// check if the GBT Header corresponds to GBT payload header
  bool isDataHeader() const { return id == GBTFlagDataHeader; }

  /// check if the GBT Header corresponds to GBT payload trailer
  bool isDataTrailer() const { return id == GBTFlagDataTrailer; }

  /// check if the GBT Header corresponds to GBT trigger word
  bool isTriggerWord() const { return id == GBTFlagTrigger; }

  /// check if the GBT Header corresponds to Diagnostic data
  bool isDiagnosticWord() const { return id == GBTFlagDiagnostic; }

  /// check if the GBT Header corresponds to Calibration word
  bool isCalibrationWord() const { return id == GBTFlagCalibration; }

  /// check if the GBT Header corresponds to ITS IB data (header is combined with lanes info)
  bool isDataIB() const { return (id & 0xe0) == GBTFlagDataIB; }

  /// check if the GBT Header corresponds to ITS IB diagnostics data (header is combined with lanes info)
  bool isCableDiagnosticIB() const { return (id & 0xe0) == GBTFlagDiagnosticIB; }

  /// check if the GBT Header corresponds to ITS OB data (header is combined with lanes/connector info)
  bool isDataOB() const { return (id & 0xe0) == GBTFlagDataOB; }

  /// check if the GBT Header corresponds to MFT data (header is combined with cable number)
  bool isDataMFT() const { return (id & 0xe0) == GBTFlagDataMFT; }

  /// check if the GBT Header corresponds to ITS OB diagnostics data (header is combined with lanes info)
  bool isCableDiagnosticOB() const { return (id & 0xe0) == GBTFlagDiagnosticIB; }

  /// check if the GBT Header corresponds to ITS IB or OB data (header is combined with lanes/connector info)
  bool isData() const { return isDataIB() || isDataOB() || isDataMFT(); }

  bool isCableDiagnostic() const { return isCableDiagnosticIB() || isCableDiagnosticIB(); }

  bool isStatus() const { return (id & 0xe0) == GBTFlagStatus; }

  const uint64_t* getW64() const { return data64; }
  const uint8_t* getW8() const { return data8; }

  uint8_t getHeader() const { return id; }

  void printX(bool padded = true) const;
  void printB(bool padded = true) const;

  ClassDefNV(GBTWord, 1);
};

struct GBTDataHeader : public GBTWord {
  /// Definition of ITS/MFT GBT Header: 80 bits long word
  /// In CRU data it must be the 1st word of the payload
  ///
  /// bits  0 : 27, Active lanes pattern
  /// bits 28 : 71, not used
  /// bits 72 : 79, header/trailer indicator

  GBTDataHeader() { id = GBTFlagDataHeader; }
  GBTDataHeader(uint32_t lanes)
  {
    id = GBTFlagDataHeader;
    activeLanes = lanes;
  }
  ClassDefNV(GBTDataHeader, 1);
};

struct GBTDataHeaderL : public GBTWord { // legacy version
  /// Definition of ITS/MFT GBT Header: 80 bits long word
  /// In CRU data it must be the 1st word of the payload
  ///
  /// bits  0 : 15, Index of GBT packet within trigger
  /// bits 16 : 43, Active lanes pattern
  /// bits 44 : 71, not used
  /// bits 72 : 79, header/trailer indicator

  GBTDataHeaderL() { id = GBTFlagDataHeader; }
  GBTDataHeaderL(int packetID, uint32_t lanes)
  {
    id = GBTFlagDataHeader;
    activeLanesL = lanes;
    packetIdx = packetID;
  }
  ClassDefNV(GBTDataHeaderL, 1);
};

struct GBTDataTrailer : public GBTWord {
  /// Definition of ITS/MFT GBT trailer: 80 bits long word
  /// In CRU data it must be the last word of the payload
  ///
  /// bits  0 : 27, Lanes stops received   // not used at the moment
  /// bits 28 : 55, Lane timeouts received // not used at the moment
  /// bits 56 : 63, not used
  /// bits 64 : 71, State of GBT_Packet:
  ///               4: lane_timeouts,  if at least 1 lane timed out
  ///               3: lane_starts_violation,  if at least 1 lane had a start violation
  ///               2: packet_overflow, if max number of packets reached
  ///               1: transmission_timeout, if timeout of transmission (lanes)
  ///               0: packet_done, if Packet finished
  enum PacketStates : int {
    PacketDone = 0,                              // Packet finished
    TransmissionTimeout = 1,                     // if timeout of transmission (lanes)
    PacketOverflow = 2,                          // if max number of packets reached
    LaneStartsViolation = 3,                     // if at least 1 lane had a start violation
    LaneTimeouts = 4,                            // if at least 1 lane timed out
    NStatesDefined,                              // total number of states defined
    MaxStateCombinations = 0x1 << NStatesDefined // number of possible combinations
  };

  GBTDataTrailer() { id = GBTFlagDataTrailer; }
  GBTDataTrailer(int lanesStop, int lanesTimeOut, bool done, bool transmTO, bool overflow, bool laneStViol, bool laneTO)
  {
    lanesStops = lanesStop;
    lanesTimeout = lanesTimeOut;
    packetDone = done;
    transmissionTimeout = transmTO;
    packetOverflow = overflow;
    laneStartsViolation = laneStViol;
    laneTimeouts = laneTO;
    id = GBTFlagDataTrailer;
  }

  uint8_t getPacketState() const { return data8[8]; }

  void setByte(uint8_t v, int which) = delete;

  ClassDefNV(GBTDataTrailer, 1);
};

struct GBTTrigger : public GBTWord {
  /// Definition of ITS/MFT Trigger status word
  /// Precedes the single trigger (continuous or triggered) data block
  ///
  /// bits  0 : 11, Trigger Type, 12 lowest bits of trigger type received from CTP for HB or 0
  /// bit  12       Internal Trigger, Used in Continuous Mode for internally generated trigger
  /// bits 13       No data expected (too close to previous trigger or error)
  /// bits 14       reserved
  /// bits 15       reserved
  /// bits 16:27    Trigger BC, HB or internal trigger BC count
  /// bits 28:31    reserved
  /// bits 32:63    Trigger Orbit, Last received HB Orbit + internal trigger orbit count
  /// bits 64:71    reserved
  /// bits 72:79    ID = 0xc0; Trigger Status Word (TrgSW) identifier

  GBTTrigger() { id = GBTFlagTrigger; }
  void setByte(uint8_t v, int which) = delete;

  ClassDefNV(GBTTrigger, 1);
};

struct GBTData : public GBTWord {
  /// Definition of ITS/MFT GBT paiload: 80 bits long word (can be padded to 128b) shipping
  /// Alpide data and having GBTHeader at positions 72:79
 public:
  /// extract lane of the IB as lowest 5 bits (ATTENTION: no check if this is really an IB header)
  static int getLaneIB(uint8_t v) { return v & 0x1f; }
  /// extract connector/lane of the OB as lowest 2/3 bits (ATTENTION: no check if this is really an OB header)
  static int getConnectorOB(uint8_t v) { return (v & 0x18) >> 3; } // lane only
  static int getLaneOB(uint8_t v) { return v & 0x7; }              // connector only
  static int getLaneOB(uint8_t v, int& connector)
  { // separately lane and connector
    connector = getConnectorOB(v);
    return getLaneOB(v);
  }
  static int getCableID(uint8_t v) { return v & 0x1f; } // combined connector and lane

  /// extract lane of the IB as lowest 5 bits (ATTENTION: no check if this is really an IB header)
  int getLaneIB() const { return getLaneIB(id); }
  /// extract connector/lane of the OB as lowest 2/3 bits (ATTENTION: no check if this is really an OB header)
  int getConnectorOB() const { return getConnectorOB(id); }                // lane only
  int getLaneOB() const { return getLaneOB(id); }                          // connector only
  int getLaneOB(int& connector) const { return getLaneOB(id, connector); } // separately lane and connector
  int getCableID() const { return getCableID(id); }                        // combined connector and lane

  ClassDefNV(GBTData, 1);
};

struct GBTDiagnostic : public GBTWord {
  /// Definition of GBT diagnostic word
  /// In CRU data it must be the only word after the RDH with stop
  ///
  /// bits  0 : 71, reserved
  /// bits 72 : 79, diagnostic flag

  GBTDiagnostic() { id = GBTFlagDiagnostic; }
  ClassDefNV(GBTDiagnostic, 1);
};

struct GBTCableDiagnostic : public GBTWord {
  /// Definition of cable diagnostic word
  ///
  /// bits  0 : 64, Error specific diagnostic data
  /// bits 63 : 71, Identifier of the specific error condition
  /// bits 72 : 79, IB or OB diagnostic flag + cable id

  GBTCableDiagnostic(bool ib = true, int lane = 0) { id = (ib ? GBTFlagDiagnosticIB : GBTFlagDiagnosticOB) | (lane & 0x1f); }
  int getCableID() const { return id & 0x1f; } // combined connector and lane
  bool isIB() const { return (id & 0xe0) == GBTFlagDiagnosticIB; }
  bool isOB() const { return (id & 0xe0) == GBTFlagDiagnosticOB; }
  ClassDefNV(GBTCableDiagnostic, 1);
};

struct GBTCableStatus : public GBTWord { // not sure this is correct, FIXME
  /// Definition of cable status word
  ///
  /// bits 72 : 79, Status flag + cable id

  GBTCableStatus(int lane = 0) { id = GBTFlagStatus | (lane & 0x1f); }
  int getCableID() const { return id & 0x1f; } // combined connector and lane
  ClassDefNV(GBTCableStatus, 1);
};

struct GBTCalibration : public GBTWord { // calibration data word
  /// bits  0 : 47, user-written tagging fields
  /// bits 48 : 71, self-incrementing counter of CDW words
  /// bits 72 : 79, calibration indicator

  GBTCalibration() { id = GBTFlagCalibration; }
  GBTCalibration(uint64_t userData, uint16_t counter = 0)
  {
    id = GBTFlagCalibration;
    calibUserField = userData & ((0x1UL << 48) - 1);
    calibCounter = counter & ((0x1 << 24) - 1);
  }
  ClassDefNV(GBTCalibration, 1);
};

} // namespace itsmft
} // namespace o2

#endif
