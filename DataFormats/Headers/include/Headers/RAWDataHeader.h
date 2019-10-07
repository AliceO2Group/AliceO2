// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_HEADER_RAWDATAHEADER_H
#define ALICEO2_HEADER_RAWDATAHEADER_H

/// @file   RAWDataHeader.h
/// @since  2017-11-22
/// @brief  Definition of the RAW Data Header

#include <cstdint>

namespace o2
{
namespace header
{

/// The definition of the RAW Data Header is specified in
/// https://docs.google.com/document/d/1IxCCa1ZRpI3J9j3KCmw2htcOLIRVVdEcO-DDPcLNFM0
/// Every 8kB readout page starts with the RDH.
///
/// Note: the definition requires little endian architecture, for the moment we
/// assume that this is the only type the software has to support (based on
/// experience with previous systems)
///
/// RAWDataHeaderV5
/// In version 5, the RDH stores the information of the heartbeat trigger
/// which opens the page. Additional detector specific triggers can be in the
/// payload.
///
/// preliminary description of the fields can be found here
/// https://gitlab.cern.ch/AliceO2Group/wp6-doc/blob/master/rdh/RDHV5.md
/// accessed on Oct 07 2019
/// FIXME: replace citation with correct ALICE note reference when published
///
///       63     56      48      40      32      24      16       8       0
///       |---------------|---------------|---------------|---------------|
///
///       |                       | priori|               |    header     |
/// 0     | reserve zero          | ty bit|    FEE id     | size  |version|
///
/// 1     |ep | cru id    |pcount|link id |  memory size  |offset nxt pack|
///
/// 2     |                orbit          | reserved         |bunch cross |
///
/// 3     |                          reserved                             |
///
/// 4     |  zero | stop  |   page count  |      trigger type             |
///
/// 5     |                          reserved                             |
///
/// 6     |      zero     | detector par  |         detector field        |
///
/// 5     |                          reserved                             |
struct RAWDataHeaderV5 {
  union {
    // default value
    uint64_t word0 = 0x00000000ffff4005;
    //                       | |     | version 5
    //                       | |   | 8x64 bit words = 64 (0x40) byte
    //                       | | invalid FEE id
    //                       | priority bit 0
    struct {
      uint64_t version : 8;        /// bit  0 to  7: header version
      uint64_t headerSize : 8;     /// bit  8 to 15: header size
      uint64_t feeId : 16;         /// bit 16 to 31: FEE identifier
      uint64_t priority : 8;       /// bit 32 to 39: priority bit
      uint64_t zero0 : 24;         /// bit 40 to 63: zeroed
    };                             ///
  };                               ///
  union {                          ///
    uint64_t word1 = 0x0;          /// data written by the CRU
    struct {                       ///
      uint32_t offsetToNext : 16;  /// bit 64 to 79:  offset to next packet in memory
      uint32_t memorySize : 16;    /// bit 80 to 95:  memory size
      uint32_t linkID : 8;         /// bit 96 to 103: link id
      uint32_t packetCounter : 8;  /// bit 104 to 111: packet counter
      uint16_t cruID : 12;         /// bit 112 to 123: CRU ID
      uint32_t endPointID : 4;     /// bit 124 to 127: DATAPATH WRAPPER ID: number used to
    };                             ///                 identify one of the 2 End Points [0/1]
  };                               ///
  union {                          ///
    uint64_t word2 = 0x0;          ///
    struct {                       ///
      uint32_t bunchCrossing : 12; /// bit 0 to 11: bunch crossing counter
      uint32_t reserved2 : 20;     /// bit 12 to 31: reserved
      uint32_t orbit;              /// bit 32 to 63: orbit
    };                             ///
  };                               ///
  union {                          ///
    uint64_t word3 = 0x0;          /// bit  0 to 63: zeroed
  };                               ///
  union {                          ///
    uint64_t word4 = 0x0;          ///
    struct {                       ///
      uint64_t triggerType : 32;   /// bit  0 to 31: trigger type
      uint64_t pageCnt : 16;       /// bit 32 to 47: pages counter
      uint64_t stop : 8;           /// bit 48 to 53: stop code
      uint64_t zero4 : 8;          /// bit 54 to 63: zeroed
    };                             ///
  };                               ///
  union {                          ///
    uint64_t word5 = 0x0;          /// bit  0 to 63: zeroed
  };                               ///
  union {                          ///
    uint64_t word6 = 0x0;          ///
    struct {                       ///
      uint64_t detectorField : 32; /// bit  0 to 31: detector field
      uint64_t detectorPAR : 16;   /// bit 32 to 47: detector PAR (Pause and Reset)
      uint64_t zero6 : 16;         /// bit 48 to 63: zeroed
    };                             ///
  };                               ///
  union {                          ///
    uint64_t word7 = 0x0;          /// bit  0 to 63: zeroed
  };
};

/// RDH v4 consists of 4 64 bit words, each of the words is extended to 128 bits
/// by the CRU
/// preliminary description of the fields can be found here
/// https://gitlab.cern.ch/AliceO2Group/wp6-doc/blob/master/rdh/RDHV4.md
/// accessed on Oct 07 2019, the old document is under
/// https://docs.google.com/document/d/1otkSDYasqpVBDnxplBI7dWNxaZohctA-bvhyrzvtLoQ
///
/// RDH v3 is identical to v4 except that a couple of fields in the CRU filled data
/// in word 1 are not defined (CRU ID, packet counter)
///
/// 32 bit words of Field 4 and 6. Fixed on Sep 17 2018 in the document.
///
///
///       63     56      48      40      32      24      16       8       0
///       |---------------|---------------|---------------|---------------|
///
///       |reserve| prior |                               |    header     |
/// 0     | zero  |ity bit|    FEE id     | block length  | size  | vers  |
///
/// 1     |ep | cru id    |pcount|link id |  memory size  |offset nxt pack|
///
/// 2     |      heartbeat orbit          |       trigger orbit           |
///
/// 3     |                          reserved                             |
///
/// 4     |      trigger type             |res|   HB BC   |res|trigger BC |
///
/// 5     |                          reserved                             |
///
/// 6     |res|    page count     | stop  | detector par  |detector field |
///
/// 5     |                          reserved                             |
///
/// Field 1,3,5,7 are reserved fields and added to extend each word to 128 bit, marked
/// grey in the documentation to indicate that those fields are added by the CRU
/// Field 1 contains additional information added by the CRU, like actual data size
/// and offset to the next page.
///
/// Field description:
/// -  8 header version: the header version number
/// -  8 header size:    the header size in byte(s)
/// - 16 block length:   assumed to be in byte, but discussion not yet finalized
/// - 16 FEE ID:         unique id of the Frontend equipment
/// -  8 priority bit:   indicates packet packet transport of higher priority
/// - 16 next package:   offset to next page
/// - 16 memory size:    actual data size in bytes, filled by CRU
/// -  8 Link ID:        set by the CRU
/// -  8 packet counter  counter increased for every packet received in the link
/// - 12 CRU ID:         number used to identify the CRU
/// -  4 DATAPATH ID:    number used to identify one of the 2 End Points [0/1]
/// - 32 trigger orbit:  trigger timing
/// - 32 heartbeat orbit: heartbeat timing
/// - 12 trigger BC:     bunch crossing parameter for trigger
/// - 12 beartbeat BC:   bunch crossing parameter for heartbeat
/// - 32 trigger type:   bit fiels for the trigger type yet to be decided
/// - 16 detector field: detector specific field
/// - 16 detector par:   detector specific field
/// -  8 stop:           bit 0 of the stop field is set if this is the last page
/// - 16 page count:     incremented if data is bigger than the page size, pages are
///                      incremented starting from 0
struct RAWDataHeaderV4 {
  union {
    // default value
    uint64_t word0 = 0x0000ffff00004004;
    //                   | | | |     | version 4
    //                   | | | |   | 8x64 bit words
    //                   | | | | block length 0
    //                   | | invalid FEE id
    //                   | priority bit 0
    struct {
      uint64_t version : 8;        /// bit  0 to  7: header version
      uint64_t headerSize : 8;     /// bit  8 to 15: header size
      uint64_t blockLength : 16;   /// bit 16 to 31: block length
      uint64_t feeId : 16;         /// bit 32 to 47: FEE identifier
      uint64_t priority : 8;       /// bit 48 to 55: priority bit
      uint64_t zero0 : 8;          /// bit 56 to 63: zeroed
    };                             ///
  };                               ///
  union {                          ///
    uint64_t word1 = 0x0;          /// data written by the CRU
    struct {                       ///
      uint32_t offsetToNext : 16;  /// bit 64 to 79:  offset to next packet in memory
      uint32_t memorySize : 16;    /// bit 80 to 95:  memory size
      uint8_t linkID : 8;          /// bit 96 to 103: link id
      uint8_t packetCounter : 8;   /// bit 104 to 111: packet counter
      uint16_t cruID : 12;         /// bit 112 to 123: CRU ID
      uint8_t endPointID : 4;      /// bit 124 to 127: DATAPATH WRAPPER ID: number used to
    };                             ///                 identify one of the 2 End Points [0/1]
  };                               ///
  union {                          ///
    uint64_t word2 = 0x0;          ///
    struct {                       ///
      uint32_t triggerOrbit;       /// bit 0 to 31: trigger orbit
      uint32_t heartbeatOrbit;     /// bit 32 to 63: trigger orbit
    };                             ///
  };                               ///
  union {                          ///
    uint64_t word3 = 0x0;          /// bit  0 to 63: zeroed
  };                               ///
  union {                          ///
    uint64_t word4 = 0x0;          ///
    struct {                       ///
      uint64_t triggerBC : 12;     /// bit 32 to 43: trigger BC ID
      uint64_t zero41 : 4;         /// bit 44 to 47: zeroed
      uint64_t heartbeatBC : 12;   /// bit 48 to 59: heartbeat BC ID
      uint64_t zero42 : 4;         /// bit 60 to 63: zeroed
      uint64_t triggerType : 32;   /// bit  0 to 31: trigger type
    };                             ///
  };                               ///
  union {                          ///
    uint64_t word5 = 0x0;          /// bit  0 to 63: zeroed
  };                               ///
  union {                          ///
    uint64_t word6 = 0x0;          ///
    struct {                       ///
      uint64_t detectorField : 16; /// bit 32 to 47: detector field
      uint64_t par : 16;           /// bit 48 to 63: par
      uint64_t stop : 8;           /// bit  0 to  7: stop code
      uint64_t pageCnt : 16;       /// bit  8 to 23: pages counter
      uint64_t zero6 : 8;          /// bit 24 to 31: zeroed
    };
  };
  union {
    uint64_t word7 = 0x0; /// bit  0 to 63: zeroed
  };
};

/// RDH v2
/// this is the version 2 definition which probably has not been adapted by any FEE
/// https://docs.google.com/document/d/1FLcBrPaF3Bg1Pnm17nwaxNlenKtEk3ocizEAiGP58J8
///
///       63     56      48      40      32      24      16       8       0
///       |---------------|---------------|---------------|---------------|
///
/// 0     | zero  |  size |link id|    FEE id     |  block length | vers  |
///
/// 1     |      heartbeat orbit          |       trigger orbit           |
///
/// 2     | zero  |heartbeatBC|      trigger type             | trigger BC|
///
/// 3     | zero  |      par      | detector field| stop  |  page count   |
///
/// Field description:
/// - version:      the header version number
/// - block length: assumed to be in byte, but discussion not yet finalized
/// - FEE ID:       unique id of the Frontend equipment
/// - Link ID:      id of the link within CRU
/// - header size:  number of 64 bit words
/// - heartbeat and trigger orbit/BC: LHC clock parameters, still under
///                 discussion whether separate fields for HB and trigger
///                 information needed
/// - trigger type: bit fiels for the trigger type yet to be decided
/// - page count:   incremented if data is bigger than the page size, pages are
///                 incremented starting from 0
/// - stop:         bit 0 of the stop field is set if this is the last page
/// - detector field and par are detector specific fields
struct RAWDataHeaderV2 {
  union {
    // default value
    uint64_t word0 = 0x0004ffffff000002;
    //                   | | |   |   | version 2
    //                   | | |   | block length 0
    //                   | | | invalid FEE id
    //                   | | invalid link id
    //                   | header size 4 x 64 bit
    struct {
      uint64_t version : 8;      /// bit 0 to 8: header version
      uint64_t blockLength : 16; /// bit 9 to 23: block length
      uint64_t feeId : 16;       /// bit 24 to 39: FEE identifier
      uint64_t linkId : 8;       /// bit 40 to 47: link identifier
      uint64_t headerSize : 8;   /// bit 48 to 55: header size
      uint64_t zero0 : 8;        /// bit 56 to 63: zeroed
    };
  };
  union {
    uint64_t word1 = 0x0;
    struct {
      uint32_t triggerOrbit;   /// bit 0 to 31: trigger orbit
      uint32_t heartbeatOrbit; /// bit 32 to 63: trigger orbit
    };
  };
  union {
    uint64_t word2 = 0x0;
    struct {
      uint64_t triggerBC : 12;   /// bit 0 to 11: trigger BC ID
      uint64_t triggerType : 32; /// bit 12 to 43: trigger type
      uint64_t heartbeatBC : 12; /// bit 44 to 55: heartbeat BC ID
      uint64_t zero2 : 8;        /// bit 56 to 63: zeroed
    };
  };
  union {
    uint64_t word3 = 0x0;
    struct {
      uint64_t pageCnt : 16;       /// bit 0 to 15: pages counter
      uint64_t stop : 8;           /// bit 13 to 23: stop code
      uint64_t detectorField : 16; /// bit 24 to 39: detector field
      uint64_t par : 16;           /// bit 40 to 55: par
      uint64_t zero3 : 8;          /// bit 56 to 63: zeroed
    };
  };
};

using RAWDataHeader = RAWDataHeaderV4;
} // namespace header
} // namespace o2

#endif // ALICEO2_HEADER_RAWDATAHEADER_H
