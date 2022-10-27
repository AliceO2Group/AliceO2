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

#ifndef ALICEO2_TRD_RAWDATA_H
#define ALICEO2_TRD_RAWDATA_H

/// \class TRDRDH
/// \brief Header for TRD raw data header
//  this is the header added by the CRU

#include <array>
#include <map>
#include <cstdint>
#include <ostream>
#include "DataFormatsTRD/Constants.h"
#include "Rtypes.h"

namespace o2
{
namespace trd
{

/// \structure HalfCRUHeader
/// \brief Header for half a cru, each cru has 2 output, 1 for each pciid.
//         This comes at the top of the data stream for each event and 1/2cru

struct HalfCRUHeader {

  /* Half cru header
64 bits is too wide, hence reduce to 32 to make it readable.
        |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
        -------------------------------------------------------------------------------------------------
Word 0  |  eventtype   | end point | stopbit|             bunchcrossing         |      headerversion    |
        -------------------------------------------------------------------------------------------------
Word 0  |                                          reserved 1                                           |
        -------------------------------------------------------------------------------------------------
Word 1  |  link 3  errorflags   |    link 2 errorflags  |  link 1 error flags   |  link 0 error flags   |
        -------------------------------------------------------------------------------------------------
Word 1  |  link 7 error flags   |   link 6 error flags  |  link 5 error flags   |  link 4 error flags   |
        -------------------------------------------------------------------------------------------------
Word 2  |  link 11 error flags  |   link 10 error flags |  link 9 error flags   |  link 8 error flags   |
        -------------------------------------------------------------------------------------------------
Word 2  |      reserved 2       |  link 14 error flags  |  link 13 error flags  |  link 12 error flags  |
         ------------------------------------------------------------------------------------------------
Word 3  |                                          reserved 3                                           |
        -------------------------------------------------------------------------------------------------
Word 3  |                                          reserved 4                                           |
        -------------------------------------------------------------------------------------------------
Word 4  |            link 1 datasize                    |             link 0 datasize                   |
        -------------------------------------------------------------------------------------------------
Word 4  |            link 3 datasize                    |             link 2 datasize                   |
        -------------------------------------------------------------------------------------------------
Word 5  |            link 5 datasize                    |             link 4 datasize                   |
        -------------------------------------------------------------------------------------------------
Word 5  |            link 7 datasize                    |             link 6 datasize                   |
        -------------------------------------------------------------------------------------------------
Word 6  |            link 9 datasize                    |             link 8 datasize                   |
        -------------------------------------------------------------------------------------------------
Word 6  |            link 11 datasize                   |             link 10 datasize                  |
        -------------------------------------------------------------------------------------------------
Word 7  |            link 13 datasize                   |             link 12 datasize                  |
        -------------------------------------------------------------------------------------------------
Word 7  |              reserved 5                       |             link 14 datasize                  |
        -------------------------------------------------------------------------------------------------
*/

  union {
    uint64_t word0 = 0x0;
    //first word          *
    //                6         5         4         3         2         1
    //             3210987654321098765432109876543210987654321098765432109876543210
    // uint64_t:   0000000000000000000000000000000000000000000000000000000000000000
    //             |                               |   |   |   |           |-------- 0..7   TRD Header version
    //             |                               |   |   |   |-------------------- 8..19  bunch crossing
    //             |                               |   |   |------------------------ 20..23 stop bit
    //             |                               |   |---------------------------- 24..27 end point
    //             |                               |-------------------------------- 28..31 event type
    //             |---------------------------------------------------------------- 32..63 reserved1
    struct {
      uint64_t HeaderVersion : 8;  // TRD Header Version
      uint64_t BunchCrossing : 12; // bunch crossing of the physics trigger.
      //NB  The BC in the RDH is the BC sent together with the heartbeat trigger, while the BC in the HalfCRUHeader is the BC of the physics trigger where the data that follows the HalfCRUHeader belongs to. However, it is not forbidden for CTP to send the heartbeat trigger together with a physics trigger, in this case the two would match (accidentally).
      uint64_t StopBit : 4;        // 8 .. 11 stop bit  0x1 if TRD packet is last data packet of trigger, else 0x0
      uint64_t EndPoint : 4;       // pci end point upper or lower 15 links of cru
      uint64_t EventType : 4;      // bit 0..7 event type of the data. Trigger bits from TTC-PON message, distinguish physics from calibration events.
      uint64_t reserveda : 32;     //
    } __attribute__((__packed__));
  };
  union {
    uint64_t word12[2];
    // 15 8 bit error flags and 1 8 bit reserved.
    struct {
      struct {
        uint8_t errorflag : 8;
      } __attribute__((__packed__)) errorflags[15];
      uint8_t reserved2 : 8;
    } __attribute__((__packed__));
  };
  union {
    uint64_t word3 = 0x0;
    struct {
      uint64_t reserved34;
    } __attribute__((__packed__));
  };
  union {
    uint64_t word47[4];
    //15 16 bit data sizes and 1 16 bit reserved word.
    struct {
      struct {
        uint64_t size : 16;
      } __attribute__((__packed__)) datasizes[15]; // although this is 8 dont use index 0 as its part of reserved.
      uint16_t reserved5;
    } __attribute__((__packed__));
  };
};

/// \structure TrackletHCHeader
/// \brief Header for each half chamber
//         Coming before all other tracklet data, a single tracklet word as a header for its half chamber

struct TrackletHCHeader {
  union {
    //             10987654321098765432109876543210
    // uint32_t:   33222222222211111111110000000000
    //                 cccccccccccccccXSSSSS   SSSy
    //             ffff|              |    |sss  ||
    //             |   |              |    |  |  ||--  0    side
    //             |   |              |    |  |  |---  1-3  stack
    //             |   |              |    |  |------  4-6  layer
    //             |   |              |    |--------  7-11 sector
    //             |   |              |------------- 12 always 1
    //             |   ----------------------------- 13-27 MCM Clock counter
    //             --------------------------------- 28-31 tracklet data format number
    uint32_t word;
    struct {
      uint32_t side : 1;  // side of chamber
      uint32_t stack : 3;
      uint32_t layer : 3;
      uint32_t supermodule : 5;
      uint32_t one : 1;   //always 1
      uint32_t MCLK : 15; // MCM clock counter 120MHz ... for simulation -- incrementing, and uniform across an event
      uint32_t format : 4;
      //  0 baseline PID 3 time slices, 7 bit each
      //  1 DO NOT USE ! reserved for tracklet end marker disambiguation
      //  14 Tracklet test-pattern mode
      //  15 Reserved for testing
    } __attribute__((__packed__));
  };
};

/// \structure TrackletMCMHeader
/// \brief Header for MCM tracklet data outuput
//         This constitutes the "4x32" bits of information from a single MCM, TrackletMCMHeader and 1-3 TrackletMCMData.
struct TrackletMCMHeader {
  //first word          *
  //             10987654321098765432109876543210
  // uint32_t:   33222222222211111111110000000000
  //             1zzzz  pppppppp        pppppppp1
  //             ||   yy|       pppppppp |      |--- 0 1 check bits
  //             ||   | |       |        ----------- 1-8   pid for cpu0 second part
  //             ||   | |       -------------------- 9-16  pid for cpu1 second part
  //             ||   | ---------------------------- 17-24 pid for cpu2 second part
  //             ||   ------------------------------ 25-26 col
  //             |---------------------------------- 27-30 padrow
  //             ----------------------------------- 31 1
  //TODO need to check endianness, I have a vague memory the trap chip has different endianness to x86.
  union {
    uint32_t word;
    struct {
      uint32_t oneb : 1;   //
      uint32_t pid0 : 8;   // part of pid calculated in cpu0 // 6 bits of Q2 and 2 bits of Q1
      uint32_t pid1 : 8;   // part of pid calculated in cpu1
      uint32_t pid2 : 8;   // part of pid calculated in cpu2
      uint32_t col : 2;    //  2 bits for position in pad direction.
      uint32_t padrow : 4; //  padrow,z coordinate for chip.
      uint32_t onea : 1;   //
    } __attribute__((__packed__));
  };
};

//  \structure TrackletMCMData.
//  \brief Raw Data of a tracklet, part is how ever in the MCM Header hence both are grouped together in the same file

struct TrackletMCMData {
  union {
    uint32_t word;
    struct {
      uint8_t checkbit : 1; //
      uint16_t slope : 8;   // Deflection angle of tracklet
      uint16_t pid : 12;    // Particle Identity 6 bits of Q0 and 6 bits of Q1
      uint16_t pos : 11;    // Position of tracklet, signed 11 bits, granularity 1/80 pad widths, -12.80 to +12.80, relative to centre of pad 10
    } __attribute__((__packed__));
  };
};

/// \structure TRDFeeID
/// \brief Frontend Electronics ID, is made up of supermodule, a/c side and the end point encoded as below.

struct TRDFeeID {
  //
  //             5432109876543210
  // uint16_t:   0000000000000000
  //             mmmmmmmm         --- supermodule 0 - 17
  //                     xxx      --- unused1
  //                        s     --- side 0=A C=1
  //                         xxx  --- unused 2;
  //                            e --- endpoint 0=lower, 1=upper
  union {
    uint16_t word;
    struct {
      uint8_t endpoint : 1;    // the pci end point of the cru in question
      uint8_t unused2 : 3;     // seperate so easier to read in hex dumps
      uint8_t side : 1;        // the A=0 or C=1 side of the supermodule being readout
      uint8_t unused1 : 3;     // seperate so easier to read in hex dumps
      uint8_t supermodule : 8; // the supermodule being read out 0-17
    } __attribute__((__packed__));
  };
};

/// \structure DigitHCHeader
/// \brief Digit version of the TrackletHCHeader above, although contents are rather different.
//  TODO come back and comment the fields or make the name more expressive, and fill in the jjjjjjj
struct DigitHCHeader {
  //
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  //
  union { // section 15.6.1 in tdp
    uint32_t word;
    struct {
      uint32_t res : 2;
      uint32_t side : 1;
      uint32_t stack : 3;
      uint32_t layer : 3;
      uint32_t supermodule : 5;
      uint32_t numberHCW : 3;
      uint32_t minor : 7;
      uint32_t major : 7;
      uint32_t version : 1;
    } __attribute__((__packed__));
  };
};
//The next hcheaders are all optional, there can be 8, we only have 3 for now.
//They can all be distinguished by their res.
struct DigitHCHeader1 {
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  //
  union { //section 15.6.2 in tdp
    uint32_t word;
    struct {
      uint32_t res : 2;
      uint32_t ptrigcount : 4;
      uint32_t ptrigphase : 4;
      uint32_t bunchcrossing : 16;
      uint32_t numtimebins : 6;
    } __attribute__((__packed__));
  };
};
struct DigitHCHeader2 {
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  union { //section 15.6.3 in tdp
    uint32_t word;
    struct {
      uint32_t res : 6;
      uint32_t dfilter : 6;
      uint32_t rfilter : 1;
      uint32_t nlfilter : 1;
      uint32_t xtfilter : 1;
      uint32_t tfilter : 1;
      uint32_t gfilter : 1;
      uint32_t pfilter : 6;
    } __attribute__((__packed__));
  };
};
struct DigitHCHeader3 {
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  union { //section 15.6.4 in tdp
    uint32_t word;
    struct {
      uint32_t res : 6;
      uint32_t svnrver : 13; //readout program svn revision
      uint32_t svnver : 13;  //assember programm svn revision
    } __attribute__((__packed__));
  };
};

struct DigitMCMHeader {
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  union {
    uint32_t word; //MCM header
    struct {
      uint32_t res : 4; // reserve 1100
      uint32_t eventcount : 20;
      uint32_t mcm : 4;
      uint32_t rob : 3;
      uint32_t yearflag : 1; //< oct2007 0,  else 1
    } __attribute__((__packed__));
  };
};

// digit mask used for the zero suppressed digit data
struct DigitMCMADCMask {
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  union {
    uint32_t word; //MCM ADC MASK header
    struct {
      uint32_t j : 4; // 0xc
      uint32_t adcmask : 21;
      uint32_t c : 5; // ~(number of bits set in adcmask)
      uint32_t n : 2; // 0b01
    } __attribute__((__packed__));
  };
};

//the odd numbering of 1 2 3 and 6 are taken from the TDP page 111 section 15.7.2, 15.7.3 15.7.4 15.7.5
struct trdTestPattern1 {
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  //  11h41 for 2flp in data stream. before that on 14th dec was1 flp in timeframe.
  union {
    uint32_t word;
    struct {
      uint32_t eventcount : 6; // lower 6 bits of e counter.
      uint32_t stack : 5;
      uint32_t layer : 3;
      uint32_t roc : 3;
      uint32_t rob : 3;
      uint32_t mcmTp2 : 4;
      uint32_t cpu : 2;
      uint32_t counter : 6;
    } __attribute__((__packed__));
  };
};

struct trdTestPattern2 {
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  union {
    uint32_t word;
    struct {
      uint32_t eventcount : 6; // lower 6 bits of e counter.
      uint32_t stack : 5;
      uint32_t layer : 3;
      uint32_t roc : 3;
      uint32_t rob : 3;
      uint32_t mcmTp2 : 4;
      uint32_t cpu : 2;
      uint32_t wordcounter : 6;
    } __attribute__((__packed__));
  };
};
struct trdTestPattern3 {
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  union {
    uint32_t word;
    struct {
      uint32_t eventcount : 12; //lower 12 bits of ecounter
      uint32_t stack : 5;
      uint32_t layer : 3;
      uint32_t roc : 3;
      uint32_t rob : 3;
      uint32_t mcm : 4;
      uint32_t cpu : 2;
    } __attribute__((__packed__));
  };
};
struct trdTestPattern6 {
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  //             1zzzz  pppppppp        pppppppp1
  union {
    uint32_t word; //HC header0
    struct {
      uint32_t eventcount; // lower 4 bits of e counter.
      uint32_t stack : 5;  // starting at 1
      uint32_t layer : 3;
      uint32_t roc : 3;
      uint32_t rob : 3;
      uint32_t mcm : 4;
      uint32_t cpu : 2;
      uint32_t wordcounter : 6;
      uint32_t oddadc : 2;
    } __attribute__((__packed__));
  };
};

struct DigitMCMData {
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  union {
    //             10987654321098765432109876543210
    // uint32_t:   00000000000000000000000000000000
    uint32_t word;
    struct {
      uint32_t f : 2;
      uint32_t z : 10;
      uint32_t y : 10;
      uint32_t x : 10;
    } __attribute__((__packed__));
  };
};

struct LinkToHCIDMapping {
  // for simplicity we store two maps to have one for each direction
  // link ID -> half-chamber ID
  // half-chamber ID -> link ID

  bool isOK() const;
  int getHCID(int link) const { return linkIDToHCID.at(link); }
  int getLink(int hcid) const { return hcIDToLinkID.at(hcid); }
  void swapLinks(int linkA, int linkB);

  std::map<int, int> linkIDToHCID;
  std::map<int, int> hcIDToLinkID;
  ClassDefNV(LinkToHCIDMapping, 1);
};

uint32_t setHalfCRUHeader(HalfCRUHeader& cruhead, int crurdhversion, int bunchcrossing, int stopbits, int endpoint, int eventtype, int feeid, int cruid);
uint32_t setHalfCRUHeaderLinkData(HalfCRUHeader& cruhead, int link, int size, int errors);
uint32_t getlinkerrorflag(const HalfCRUHeader& cruhead, const uint32_t link);
uint32_t getlinkdatasize(const HalfCRUHeader& cruhead, const uint32_t link);
uint32_t getlinkerrorflags(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linkerrorflags);
uint32_t getlinkdatasizes(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linksizes);
bool halfCRUHeaderSanityCheck(const o2::trd::HalfCRUHeader& header);
void printDigitHCHeader(o2::trd::DigitHCHeader& header, uint32_t headers[3]);

//functions updated/checked/new for new raw reader.
//above methods left for cross checking what changes have occured.
void constructTrackletHCHeader(TrackletHCHeader& header, int hcid, int chipclock, int format);
uint16_t constructTRDFeeID(int supermodule, int side, int endpoint);
uint32_t setHalfCRUHeaderFirstWord(HalfCRUHeader& cruhead, int crurdhversion, int bunchcrossing, int stopbits, int endpoint, int eventtype, int feeid, int cruid);
void setHalfCRUHeaderLinkSizeAndFlags(HalfCRUHeader& cruhead, int link, int size, int errors);
DigitMCMADCMask constructBlankADCMask();

uint32_t getHalfCRULinkInfo(const HalfCRUHeader& cruhead, const uint32_t link, const bool data);
uint32_t getHalfCRULinkErrorFlag(const HalfCRUHeader& cruhead, const uint32_t link);
uint32_t getHalfCRULinkDataSize(const HalfCRUHeader& cruhead, const uint32_t link);
void getHalfCRULinkErrorFlags(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linkerrorflags);
void getHalfCRULinkDataSizes(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linksizes);
std::ostream& operator<<(std::ostream& stream, const TrackletHCHeader& halfchamberheader);
std::ostream& operator<<(std::ostream& stream, const TrackletMCMHeader& tracklmcmhead);
std::ostream& operator<<(std::ostream& stream, const TrackletMCMData& trackletmcmdata);
std::ostream& operator<<(std::ostream& stream, const DigitHCHeader& halfchamberheader);
std::ostream& operator<<(std::ostream& stream, const DigitMCMHeader& digitmcmhead);
std::ostream& operator<<(std::ostream& stream, const DigitMCMData& digitmcmdata);
std::ostream& operator<<(std::ostream& stream, const DigitMCMADCMask& adcmask);
std::ostream& operator<<(std::ostream& stream, const HalfCRUHeader& halfcru);

void printTrackletHCHeader(const o2::trd::TrackletHCHeader& tracklet);
void printTrackletMCMData(const o2::trd::TrackletMCMData& tracklet);
void printTrackletMCMHeader(const o2::trd::TrackletMCMHeader& mcmhead);

void printDigitMCMData(const o2::trd::DigitMCMData& digitmcmdata);
void printDigitMCMHeader(const o2::trd::DigitMCMHeader& digitmcmhead);
void printDigitMCMADCMask(const o2::trd::DigitMCMADCMask& digitmcmadcmask);

void printHalfCRUHeader(const o2::trd::HalfCRUHeader& halfcru);
void clearHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru);
bool sanityCheckTrackletHCHeader(const o2::trd::TrackletHCHeader& header);
bool sanityCheckTrackletMCMHeader(const o2::trd::TrackletMCMHeader& header);
bool sanityCheckDigitMCMHeader(const o2::trd::DigitMCMHeader& header);
bool sanityCheckDigitMCMADCMask(const o2::trd::DigitMCMADCMask& mask);
void incrementADCMask(DigitMCMADCMask& mask, int channel);
int getDigitHCHeaderWordType(uint32_t word);
void printDigitHCHeaders(o2::trd::DigitHCHeader& header, uint32_t headers[3], int index, int offset, bool good);
void printDigitHCHeader(o2::trd::DigitHCHeader& header, uint32_t headers[3]);
}
}
#endif
