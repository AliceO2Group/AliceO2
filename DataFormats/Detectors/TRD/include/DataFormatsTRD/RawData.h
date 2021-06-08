// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_RAWDATA_H
#define ALICEO2_TRD_RAWDATA_H

#include "fairlogger/Logger.h"

/// \class TRDRDH
/// \brief Header for TRD raw data header
//  this is the header added by the CRU

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
Word 2  |  link 12 error flags  |  link 13 error flags  |  link 14 error flags  |      reserved 2       |
        -------------------------------------------------------------------------------------------------
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
      uint64_t StopBit : 4;        // 8 .. 11 stop bit  0x1 if TRD packet is last data packet of trigger, else 0x0  TODO why 4 bits if only using 1?
      uint64_t EndPoint : 4;       // bit 0..7 event type of the data. Trigger bits from TTC-PON message, distinguish physics from calibration events.
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
    //                 cccccccccccccccX LLL   SSSSS
    //             ffff|              |y|  sss|
    //             |   |              |||  |  |-----  0-4  supermodule
    //             |   |              |||  |--------  5-7  stack
    //             |   |              ||------------  8-10 layer
    //             |   |              |------------- 11    always 0x1
    //             |   |              |------------- 12 side of chamber
    //             |   ----------------------------- 13-27 MCM Clock counter
    //             --------------------------------- 28-31 tracklet data format number
    uint32_t word;
    struct {
      uint32_t supermodule : 5;
      uint32_t stack : 3;
      uint32_t layer : 3;
      uint32_t one : 1;   //always 1
      uint32_t side : 1;  // side of chamber
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
  //             ||   | |       |        ----------- 1-8   pid for tracklet 3 second part
  //             ||   | |       -------------------- 9-16  pid for tracklet 2 second part
  //             ||   | ---------------------------- 17-24 pid for tracklet 1 second part
  //             ||   ------------------------------ 25-26 col
  //             |---------------------------------- 27-30 padrow
  //             ----------------------------------- 31 1
  //TODO need to check endianness, I have a vague memory the trap chip has different endianness to x86.
  union {
    uint32_t word;
    struct {
      uint32_t oneb : 1;   //
      uint32_t pid0 : 8;   // part of pid for tracklet 0
      uint32_t pid1 : 8;   // part of pid for tracklet 1
      uint32_t pid2 : 8;   // part of pid for tracklet 2
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
      uint16_t pid : 12;    // Particle Identity
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
    uint32_t word0;
    struct {
      uint32_t res0 : 2;
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

  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  //
  union { //section 15.6.2 in tdp
    uint32_t word1;
    struct {
      uint32_t res1 : 2;
      uint32_t ptrigcount : 4;
      uint32_t ptrigphase : 4;
      uint32_t bunchcrossing : 16;
      uint32_t numtimebins : 6;
    } __attribute__((__packed__));
  };
#ifdef DIGITALHCOPTIONALHEADER
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  union { //section 15.6.3 in tdp
    uint32_t word2;
    struct {
      uint32_t res2 : 6;
      uint32_t dfilter : 6;
      uint32_t rfilter : 1;
      uint32_t nlfilter : 1;
      uint32_t xtfilter : 1;
      uint32_t tfilter : 1;
      uint32_t gfilter : 1;
      uint32_t pfilter : 6;
    } __attribute__((__packed__));
  };
  //             10987654321098765432109876543210
  // uint32_t:   00000000000000000000000000000000
  union { //section 15.6.4 in tdp
    uint32_t word3;
    struct {
      uint32_t res3 : 6;
      uint32_t svnrver : 13; //readout program svn revision
      uint32_t svnver : 13;  //assember programm svn revision
    } __attribute__((__packed__));
  };
#endif
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
      uint32_t n : 2; // unused always 0x3
      uint32_t c : 5; // unused always 0x1f
      uint32_t adcmask : 21;
      uint32_t j : 4; // unused always 0xc
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
  /*  union {
    uint32_t word0;
    struct {
      uint32_t a : 2;
      uint32_t b : 5;
      uint32_t adc : 21; //adc bit patternpad plane
    } __attribute__((__packed__));
  };*/
  union {
    //             10987654321098765432109876543210
    // uint32_t:   00000000000000000000000000000000
    uint32_t word;
    struct {
      uint32_t c : 2; // c is wrong I cant remember name, but not a concern at the moment.
      uint32_t z : 10;
      uint32_t y : 10;
      uint32_t x : 10;
    } __attribute__((__packed__));
  };
};

void buildTrackletHCHeader(TrackletHCHeader& header, int sector, int stack, int layer, int side, int chipclock, int format);
void buildTrackletHCHeaderd(TrackletHCHeader& header, int detector, int rob, int chipclock, int format);
uint16_t buildTRDFeeID(int supermodule, int side, int endpoint);
uint32_t setHalfCRUHeader(HalfCRUHeader& cruhead, int crurdhversion, int bunchcrossing, int stopbits, int endpoint, int eventtype, int feeid, int cruid);
uint32_t setHalfCRUHeaderLinkData(HalfCRUHeader& cruhead, int link, int size, int errors);
void buildTrackletMCMData(TrackletMCMData& trackletword, const uint slope, const uint pos, const uint q0, const uint q1, const uint q2);
uint32_t unpacklinkinfo(const HalfCRUHeader& cruhead, const uint32_t link, const bool data);
uint32_t getlinkerrorflag(const HalfCRUHeader& cruhead, const uint32_t link);
uint32_t getlinkdatasize(const HalfCRUHeader& cruhead, const uint32_t link);
uint32_t getlinkerrorflags(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linkerrorflags);
uint32_t getlinkdatasizes(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linksizes);
uint32_t getQFromRaw(const o2::trd::TrackletMCMHeader* header, const o2::trd::TrackletMCMData* data, int pidindex, int trackletindex);
uint32_t getHCIDFromTrackletHCHeader(const TrackletHCHeader& header);
uint32_t getHCIDFromTrackletHCHeader(const uint32_t& headerword);
std::ostream& operator<<(std::ostream& stream, const TrackletHCHeader& halfchamberheader);
std::ostream& operator<<(std::ostream& stream, const TrackletMCMHeader& mcmhead);
std::ostream& operator<<(std::ostream& stream, const TrackletMCMData& tracklet);
void printTrackletMCMData(o2::trd::TrackletMCMData& tracklet);
void printTrackletMCMHeader(o2::trd::TrackletMCMHeader& mcmhead);
void printHalfChamber(o2::trd::TrackletHCHeader& halfchamber);
void dumpHalfChamber(o2::trd::TrackletHCHeader& halfchamber);
void printHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru);
void dumpHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru);
void clearHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru);
std::ostream& operator<<(std::ostream& stream, const HalfCRUHeader& halfcru);
bool trackletMCMHeaderSanityCheck(o2::trd::TrackletMCMHeader& header);
bool trackletHCHeaderSanityCheck(o2::trd::TrackletHCHeader& header);
bool digitMCMHeaderSanityCheck(o2::trd::DigitMCMHeader* header);
void printDigitMCMHeader(o2::trd::DigitMCMHeader& header);
void printDigitHCHeader(o2::trd::DigitHCHeader& header);
DigitMCMADCMask buildBlankADCMask();
int getNumberofTracklets(o2::trd::TrackletMCMHeader& header);
void setNumberOfTrackletsInHeader(o2::trd::TrackletMCMHeader& header, int numberoftracklets);
int nextmcmadc(unsigned int& bp, int channel);
}
}
#endif
