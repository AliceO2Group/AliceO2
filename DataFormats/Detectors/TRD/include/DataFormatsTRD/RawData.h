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
      uint64_t BunchCrossing : 12; // bunch crossing
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
    // uint32_t:   00000000000000000000000000000000
    //                 cccccccccccccccc iiiiiiiiiii
    //             ffff|               y|
    //             |   |               |------       0-10 half chamber id
    //             |   |               ------------- 11 always 0x1
    //             |   ----------------------------- 12-72 MCM Clock counter
    //             --------------------------------- 28-31 tracklet data format number
    uint32_t word;

    struct {
      uint32_t HCID : 11; // half chamber id 0:1079
      uint32_t one : 1;   //always 1
      uint32_t MCLK : 16; // MCM clock counter 120MHz ... for simulation -- incrementing, and same number in all for each event.
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
  // uint32_t:   00000000000000000000000000000000
  //             1zzzz  pppppppp        pppppppp1
  //             ||   yy|       pppppppp |      |--- 0 1 check bits
  //             ||   | |       |        ----------- 1-8   pid for tracklet 3 second part
  //             ||   | |       -------------------- 9-16  pid for tracklet 2 second part
  //             ||   | ---------------------------- 17-24 pid for tracklet 1 second part
  //             ||   ------------------------------ 25-26 coly
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

struct TrackletMCMData { // This is a bad name as part of the tracklet data is in the MCMHeader.
  union {
    uint32_t word;
    struct {
      uint8_t checkbit : 1; //
      uint16_t slope : 6;   // Deflection angle of tracklet
      uint16_t pid : 15;    // Particle Identity
      uint16_t pos : 10;    // Position of tracklet, signed 10 bits, granularity 0.02 pad widths, -10.22 to +10.22, relative to centre of pad 10
    } __attribute__((__packed__));
  };
};

/*
std::vector<uint32_t> getPID(char *rawdata, int tracklet)
{
    // extract the 2 parts of the pid and return the combined PID
    // rawdata starts with the mcmheader
    uint32_t pid = 1;//(rawdata.pid[tracklet]<<15) + rawdata.pid[]; TODO figure out a better way that is not *undefined* c++ to progress to following 32bit words after TrackletMCMHeader.
    //TODO come back here, marker to come back.
    std::vector<uint32_t> pids;
    TrackletMCMHeader mcmheader;
    TrackletMCMData trackletdata;
    //memcpy(&mcmheader,&rawdata[0],sizeof(mcmheader));
    std::copy(rawdata.begin(), rawdata.begin()+sizeof(mcmheader),(char*)&mcmheader);
    for(int tracklet=0;tracklet<3;tracklet++){
        memcpy(&trackletdata,&rawdata[0]+sizeof(mcmheader)+tracklet*sizeof(trackletdata),sizeof(TrackletMCMData));
        uint32_t headpid=0;
        switch(tracklet){
            case 0 : headpid=mcmheader.pid0;break;
            case 1 : headpid=mcmheader.pid1;break;
            case 2 : headpid=mcmheader.pid2;break;
        }
        pids[tracklet]  = (headpid<<15) + trackletdata.pid;
    }
    //    memcpy(rawdata,)
    return pids;
}

uint32_t getPadRow(const char *rawdata)
{
    TrackletMCMHeader mcmheader;
    memcpy(&mcmheader, rawdata,sizeof(TrackletMCMHeader));
    return mcmheader.padrow;

}
uint32_t getCol(const char* rawdata)
{
    TrackletMCMHeader mcmheader;
    memcpy(&mcmheader, rawdata,sizeof(TrackletMCMHeader));
    return mcmheader.col;

}
int16_t getPos(const char* rawdata, int tracklet) 
{
    // extract y from the tracklet word, raw data points to the mcmheader of this data.
    //rawdata points to the TrackletMCMHeader
     TrackletMCMData trackletdata;
    TrackletMCMHeader mcmrawdataheader;
    memcpy(&mcmrawdataheader,rawdata,sizeof(TrackletMCMHeader));
    memcpy(&trackletdata,rawdata+sizeof(TrackletMCMHeader));
    return trackletdata.pos;
    return 1;
}
uint32_t getSlope(const *rawdata, int tracklet)
{
    // extract dy or slope from the tracklet word, raw data points to the mcmheader of this data.
    TrackletMCMData trackletdata;
    memcpy(&trackletdata,&rawdata[0]+sizeof(TrackletMCMHeader)+tracklet*sizeof(trackletdata),sizeof(TrackletMCMData));
    return trackletdata.slope;
}
due to the interplay of TrackletMCMHeader and TrackletMCMData come back to this. TODO
*/

uint32_t unpacklinkinfo(const HalfCRUHeader& cruhead, const uint32_t link, const bool data);
uint32_t getlinkerrorflag(const HalfCRUHeader& cruhead, const uint32_t link);
uint32_t getlinkdatasize(const HalfCRUHeader& cruhead, const uint32_t link);
uint32_t getlinkerrorflags(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linkerrorflags);
uint32_t getlinkdatasizes(const HalfCRUHeader& cruheader, std::array<uint32_t, 15>& linksizes);
std::ostream& operator<<(std::ostream& stream, const TrackletHCHeader halfchamberheader);
std::ostream& operator<<(std::ostream& stream, const TrackletMCMData& tracklet);
void printTrackletMCMData(o2::trd::TrackletMCMData& tracklet);
void printTrackletMCMHeader(o2::trd::TrackletMCMHeader& mcmhead);
std::ostream& operator<<(std::ostream& stream, const TrackletMCMHeader& mcmhead);
void printHalfChamber(o2::trd::TrackletHCHeader& halfchamber);
void dumpHalfChamber(o2::trd::TrackletHCHeader& halfchamber);
void printHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru);
void dumpHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru);
std::ostream& operator<<(std::ostream& stream, const HalfCRUHeader& halfcru);
}
}
#endif
