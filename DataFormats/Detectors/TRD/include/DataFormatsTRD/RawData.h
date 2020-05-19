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

#include <iosfwd>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "fairlogger/Logger.h"
/// \class TRDRDH
/// \brief Header for TRD raw data header 
//  this is the header added by the CRU

namespace o2{
namespace trd{

struct HalfCRUHeader
{
/* Half cru header
64 bits is too wide, hence reduce to 32 to make it readable.

        |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
        -------------------------------------------------------------------------------------------------
Word 0  |  link 0  errorflags   |               link 0 datasize                 |  link 1  errorflags   |
        -------------------------------------------------------------------------------------------------
Word 0  |   link 0 data size                            |   link 2 error flags  |  link 2 datasize upper|
        -------------------------------------------------------------------------------------------------
Word 1  | link 2 datasize lower |   link 3 errorflags   |              link 3 datasize                  |
        -------------------------------------------------------------------------------------------------
Word 1  |  link 4  errorflags   |               link 4 datasize                 |  link 5  errorflags   |
        -------------------------------------------------------------------------------------------------
Word 2  |   link 5 data size                            |   link 6 error flags  |  link 6 datasize upper|
        -------------------------------------------------------------------------------------------------
Word 2  | link 6 datasize lower |   link 7 errorflags   |              link 7 datasize                  |
        -------------------------------------------------------------------------------------------------
Word 3  |                                          reserved                                             |
        -------------------------------------------------------------------------------------------------
Word 3  | header version        | bunch crossing                    | stop bit  |  event type           |
        -------------------------------------------------------------------------------------------------
Word 4  |                                          reserved                                             |
        -------------------------------------------------------------------------------------------------
Word 4  |                                          reserved                                             |
        -------------------------------------------------------------------------------------------------
Word 5  |                      reserved                                         | link 8 error flags    |
        -------------------------------------------------------------------------------------------------
Word 5  |   link 8 data size                            |   link 9 error flags  |  link 9 datasize upper|
        -------------------------------------------------------------------------------------------------
Word 6  | link 9 datasize lower |   link 10 errorflags  |              link 10 datasize                 |
        -------------------------------------------------------------------------------------------------
Word 6  |  link 11 errorflags   |               link 11 datasize                |  link 12 errorflags   |
        -------------------------------------------------------------------------------------------------
Word 7  |   link 12 data size                            | link 13 error flags  | link 13 datasize upper|
        -------------------------------------------------------------------------------------------------
Word 7  |link 13 datasize lower |   link 14 errorflags   |              link 14 datasize                |
        -------------------------------------------------------------------------------------------------

alternate candidate :
        |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
        -------------------------------------------------------------------------------------------------
Word 0  | header version        | bunch crossing                    | stop bit  |  event type           |
        -------------------------------------------------------------------------------------------------
Word 0  |                                          reserved                                             |
        -------------------------------------------------------------------------------------------------
Word 1  |  link 0  errorflags   |    link 1 errorflags  | link2 error flags     | link 3 error flags    |
        -------------------------------------------------------------------------------------------------
Word 1  |   link 4 error flags  |   link 5 error flags  |  link 6 error flags   | link 7 error flags    |
        -------------------------------------------------------------------------------------------------
Word 2  |            link 0 datasize                    |            link 1 datasize                    |
        -------------------------------------------------------------------------------------------------
Word 2  |            link 2 datasize                    |            link 3 datasize                    |
        -------------------------------------------------------------------------------------------------
Word 3  |            link 4 data size                   |            link 5 datasize                    |
        -------------------------------------------------------------------------------------------------
Word 3  |             link 6 datasize                   |            link 7 datasize                    |
        -------------------------------------------------------------------------------------------------
Word 4  |                                          reserved                                             |
        -------------------------------------------------------------------------------------------------
Word 4  |                                          reserved                                             |
        -------------------------------------------------------------------------------------------------
Word 5  |                                         reserved                      |  link 8 error flags   |
        -------------------------------------------------------------------------------------------------
Word 5  |   link 9  errorflags  |    link 10 errorflags |   link11 error flags | link 12 error flags    |
        -------------------------------------------------------------------------------------------------
Word 6  |   link 13 error flags |   link 14 error flags |            link 8 datasize                    |
        -------------------------------------------------------------------------------------------------
Word 6  |            link 9 datasize                    |            link 10 datasize                   |
        -------------------------------------------------------------------------------------------------
Word 7  |            link 11 datasize                   |            link 12 datasize                   |
        -------------------------------------------------------------------------------------------------
Word 7  |            link 13 datasize                   |            link 14 datasize                   |
       --------------------------------------------------------------------------------------------------
*/

 union{
    uint64_t word02[3];
     struct{
        uint64_t errorflags : 8;
        uint64_t size : 16;
    }__attribute__((__packed__)) linksA[8];
 };
 union{
    uint64_t word3=0x0;
    //first word          *   
    // uint64_t: 0x0000000000000000
    //             |      - |   ||
    //             |      | |   |- 0..7 Event type 
    //             |      | |   -- 8..11 Stop bit
    //             |      | ------ 12..23 bunch crossing id 
    //             |      -------- 24..31 TRD Header version 
    //             --------------- 32..63 reserveda
    struct{
        uint64_t reserveda :32; // 
        uint64_t HeaderVersion : 8; // TRD Header Version
        uint64_t BunchCrossing :12;  // bunch crossing
        uint64_t StopBit : 4; // 8 .. 11 stop bit  0x1 if TRD packet is last data packet of trigger, else 0x0  TODO why 4 bits if only using 1?
        uint64_t EventType : 8; // bit 0..7 event type of the data. Trigger bits from TTC-PON message, distinguish physics from calibration events.
 }__attribute__((__packed__));
    };
 union{
     uint64_t word4=0x0;
     struct{
         uint64_t reservedb;
     }__attribute__((__packed__));
 };
 union{
     uint64_t word57[3];
     struct{
         uint64_t errorflags : 8;
         uint64_t size : 16;
     }__attribute__((__packed__)) linksB[8]; // although this is 8 dont use index 0 as its part of reserved.
 };
};

/// \structure HalfChamberHeader
/// \brief Header for each half chamber
//         Coming before all other tracklet data, a single tracklet word as a header for its half chamber

struct HalfChamberHeader
{
    union{
        //             10987654321098765432109876543210
        // uint32_t:   00000000000000000000000000000000
        //                 cccccccccccccccc iiiiiiiiiii
        //             ffff|               y|
        //             |   |               |------       0-10 half chamber id
        //             |   |               ------------- 11 always 0x1 
        //             |   ----------------------------- 12-72 MCM Clock counter
        //             --------------------------------- 28-31 tracklet data format number
        uint32_t word;

        struct{
            uint32_t HCID : 11; // half chamber id 0:1079
            uint32_t one : 1 ; //always 1
            uint32_t MCLK : 16 ; // MCM clock counter 120MHz ... for simulation -- incrementing, and same number in all for each event.
            uint32_t format : 4;
            //  0 baseline PID 3 time slices, 7 bit each
            //  1 DO NOT USE ! reserved for tracklet end marker disambiguation
            //  14 Tracklet test-pattern mode
            //  15 Reserved for testing
        }__attribute__((__packed__));
    };
};

/// \structure MCMRawDataHeader
/// \brief Header for MCM tracklet data outuput
//         This constitutes the "4x32" bits of information from a single MCM, MCMRawDataHeader and 1-3 TrapRawTracklet.
struct MCMRawDataHeader
{
    //first word          *   
    //             10987654321098765432109876543210
    // uint32_t:   00000000000000000000000000000000
    //             1zzzz  pppppppp       pppppppp  1
    //             ||   yy|      pppppppp |      ccc-- 0..1 check bits
    //             ||   | |      |        ------------ 2-9   pid for tracklet 3 second part
    //             ||   | |      --------------------- 10-17 pid for tracklet 2 second part
    //             ||   | ---------------------------- 18-25 pid for tracklet 1 second part
    //             ||   ------------------------------ 26-27 coly 
    //             |---------------------------------- 28-31 padrow
    //TODO need to check endianness, I have a vague memory the trap chip has different endianness to x86.
    union{
        uint32_t word;
        struct{ 
            uint32_t oneb : 1; //
            uint32_t pid0:8 ; // 3 parts of pid for each tracklet
            uint32_t pid1:8 ; // 3 parts of pid for each tracklet
            uint32_t pid2:8 ; // 3 parts of pid for each tracklet
            uint32_t col : 2; //  2 bits for position in pad direction.
            uint32_t padrow : 4; //  padrow,z coordinate for chip.
            uint32_t onea : 1; //
        }__attribute__((__packed__));
    };

};

//  \structure TrapRawTracklet.
//  \brief Raw Data of a tracklet, part is how ever in the MCM Header hence both are grouped together in the same file

 struct TrapRawTracklet{ // This is a bad name as part of the tracklet data is in the MCMHeader.
   union{
       uint32_t word;
       struct{
           uint8_t checkbit : 1;  // The size of the data for this link
           uint16_t pid : 15 ;  // Particle Identity
           uint16_t slope : 6 ;  // Deflection angle of tracklet
           uint16_t pos : 10 ;  // Position of tracklet, signed 10 bits, granularity 0.02 pad widths, -10.22 to +10.22, relative to centre of pad 10 
       }__attribute__((__packed__));
   };
 };



/*
std::vector<uint32_t> getPID(char *rawdata, int tracklet)
{
    // extract the 2 parts of the pid and return the combined PID
    // rawdata starts with the mcmheader
    uint32_t pid = 1;//(rawdata.pid[tracklet]<<15) + rawdata.pid[]; TODO figure out a better way that is not *undefined* c++ to progress to following 32bit words after MCMRawDataHeader.
    //TODO come back here, marker to come back.
    std::vector<uint32_t> pids;
    MCMRawDataHeader mcmheader;
    TrapRawTracklet trackletdata;
    //memcpy(&mcmheader,&rawdata[0],sizeof(mcmheader));
    std::copy(rawdata.begin(), rawdata.begin()+sizeof(mcmheader),(char*)&mcmheader);
    for(int tracklet=0;tracklet<3;tracklet++){
        memcpy(&trackletdata,&rawdata[0]+sizeof(mcmheader)+tracklet*sizeof(trackletdata),sizeof(TrapRawTracklet));
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
    MCMRawDataHeader mcmheader;
    memcpy(&mcmheader, rawdata,sizeof(MCMRawDataHeader));
    return mcmheader.padrow;

}
uint32_t getCol(const char* rawdata)
{
    MCMRawDataHeader mcmheader;
    memcpy(&mcmheader, rawdata,sizeof(MCMRawDataHeader));
    return mcmheader.col;

}
int16_t getPos(const char* rawdata, int tracklet) 
{
    // extract y from the tracklet word, raw data points to the mcmheader of this data.
    //rawdata points to the MCMRawDataHeader
     TrapRawTracklet trackletdata;
    MCMRawDataHeader mcmrawdataheader;
    memcpy(&mcmrawdataheader,rawdata,sizeof(MCMRawDataHeader));
    memcpy(&trackletdata,rawdata+sizeof(MCMRawDataHeader));
    return trackletdata.pos;
    return 1;
}
uint32_t getSlope(const *rawdata, int tracklet)
{
    // extract dy or slope from the tracklet word, raw data points to the mcmheader of this data.
    TrapRawTracklet trackletdata;
    memcpy(&trackletdata,&rawdata[0]+sizeof(MCMRawDataHeader)+tracklet*sizeof(trackletdata),sizeof(TrapRawTracklet));
    return trackletdata.slope;
}
due to the interplay of MCMRawDataHeader and TrapRawTracklet come back to this. TODO
*/


uint32_t unpacklinkinfo(const HalfCRUHeader& cruhead, const uint32_t link, const bool data=true)
{
    //TODO this needs some more ellaborate testing. I got the packing to work so for now, 
    //     its a question of unpacking info in line 88 and 89.
    uint32_t info=0;
    if(link>14) return 0xffffffff; 

    char *words;
    if(link<8) words = (char*)&cruhead.word02[0];
    if(link<15) words= (char*)&cruhead.word57[0];
    uint32_t byteoffset=link*3; // each link [size+erroflags] is 24 bits, so 3 bytes.
    if(data) info = ((words[byteoffset+1] )<< 8 )+ (words[byteoffset+2]);
    else info=words[byteoffset];

    return info;
}

uint32_t getlinkerrorflag(const HalfCRUHeader& cruhead, const uint32_t link)
{
    uint32_t errorflags=0;
    if(link<8){
        //dealing with word0-2
        errorflags= cruhead.linksA[link].errorflags;
    }
    else {
        if(link<16){
            errorflags=cruhead.linksB[link-8+1].errorflags; // link 0 [actually 9] is in fact the end part of reserved.

        }
        else std::cout << "error link=" << link << " not in range 0-14" << std::endl;
    }
    return errorflags;
}

uint32_t getlinkdatasize(const HalfCRUHeader& cruhead, const uint32_t link)
{
    //return number 32 byte blocks for the link 3x64bit ints.
    uint32_t size=0;
    if(link<8){
        size= (cruhead.linksA[link].size);
    }
    else {
        if(link<16){ // link 0 is part of reserved
            size=cruhead.linksB[link-8+1].size;

        }
        else std::cout << "error link=" << link << " not in range 0-14" << std::endl;
    }
    return size;
}

uint32_t getlinkerrorflags(const HalfCRUHeader& cruheader, std::array<uint32_t,15>& linkerrorflags )
{
    for(uint32_t link=0;link<15;link++){
        linkerrorflags[link] = getlinkerrorflag(cruheader,link);
    }
    return 0;
}
uint32_t getlinkdatasizes(const HalfCRUHeader& cruheader, std::array<uint32_t,15>& linksizes )
{
    for(uint32_t link=0;link<15;link++){
        linksizes[link] = getlinkdatasize(cruheader,link);
    }
    return 0;
}

/*
 *  Printing methods to dump and display the various structures above in pretty format or hexdump
 */

std::ostream& operator<<(std::ostream& stream, const HalfChamberHeader halfchamberheader)
{
    stream << "HalfChamberHeader : " << halfchamberheader.format << " ;; " << halfchamberheader.MCLK << " :: " << halfchamberheader.one << " :: " << halfchamberheader.HCID << std::endl;
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const TrapRawTracklet& tracklet)
{
    // make a pretty output of the tracklet.
    stream << "TrapRawTracklet: pos=" << tracklet.pos << "::slope=" << tracklet.slope <<"::pid="<< tracklet.pid <<"::checkbit=" << tracklet.checkbit << std::endl;
    return stream;
}
void printTrapRawTracklet(o2::trd::TrapRawTracklet& tracklet)
{
    LOGF(INFO,"TrapRawTracklet: ");
}

void printMCMHeader(o2::trd::MCMRawDataHeader& mcmhead)
{
    LOGF(INFO,"MCMRawHeader: 1:%d padrow: 0x%02x col: 0x%01x pid2 0x%02x pid1: 0x%02x pid0: 0x%02x 1:%d",mcmhead.onea,mcmhead.padrow,mcmhead.col,mcmhead.pid2,mcmhead.pid1,mcmhead.pid0,mcmhead.oneb);
}

std::ostream& operator<<(std::ostream& stream, const MCMRawDataHeader& mcmhead) 
{
    // make a pretty output of the mcm header.
    // stream << "MCMRawHeader:" << mcmhead.checkbits << "::" << (mcmhead.pid&0xfff000000) << ":"<<  (mcmhead.pid&0xfff000) << ":"<< (mcmhead.pid&0xfff) << std::endl;
    stream << "MCMRawHeader:" << mcmhead.onea<<"::"<< mcmhead.pid2 << ":"<<  mcmhead.pid1 << ":"<< mcmhead.pid0 << "::"<<mcmhead.oneb<< std::endl;
    return stream;
}


void printHalfChamber(o2::trd::HalfChamberHeader& halfchamber)
{
   LOGF(INFO,"HCID : %d MCLK: %d Format: %d Always1:%d",halfchamber.HCID,halfchamber.MCLK,halfchamber.format, halfchamber.one);
}

void dumpHalfChamber(o2::trd::HalfChamberHeader& halfchamber)
{
   LOGF(INFO,"HalfChamber : 0x%08x",halfchamber.word);
}

void printHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
    std::array<uint32_t,15> sizes;
    std::array<uint32_t,15> errorflags;
    getlinkdatasizes(halfcru,sizes);
    getlinkerrorflags(halfcru,errorflags);
    LOGF(INFO, "V:%d BC:%d SB:%d EType:%d",halfcru.HeaderVersion,halfcru.BunchCrossing,halfcru.StopBit,halfcru.EventType);
    for(int i=0;i<15;i++) LOGF(INFO,"Link %d size: %ul eflag: 0x%02x",i,sizes[i],errorflags[i]);
}

void dumpHalfCRUHeader(o2::trd::HalfCRUHeader& halfcru)
{
    std::array<uint32_t,16> raw{};
    memcpy(&raw[0],&halfcru,sizeof(halfcru));
    for(int i=0;i<4;i++){
        int index=4*i;
        LOGF(INFO,"[1/2CRUHeader %d] 0x%08x 0x%08x 0x%08x 0x%08x",i,raw[index+3],raw[index+2],raw[index+1],raw[index+0]);
    }
}


std::ostream& operator<<(std::ostream& stream, const HalfCRUHeader& halfcru) // make a pretty output of the header.
{
    stream << std::hex;
    stream << "EventType : " << halfcru.EventType << std::endl;
    stream << "StopBit : " << halfcru.StopBit << std::endl;
    stream << "BunchCrossing : " << halfcru.BunchCrossing << std::endl;
    stream << "HeaderVersion : " << halfcru.HeaderVersion << std::endl;
    stream << "link  sizes : ";
    for(int link=0;link<15;link++)stream << link<<":"<<std::hex << std::setw(4) <<  getlinkdatasize(halfcru,link) <<",";
    stream << std::endl;
    stream << "link  errorflags : ";
    for(int link=0;link<15;link++)stream << link<<":"<<std::hex << std::setw(2) << getlinkerrorflag(halfcru,link) <<",";
    stream << std::endl;
    stream << "0x" << halfcru.word02[0] << " 0x" << halfcru.word02[1] << " 0x" << halfcru.word02[2] << " 0x" << halfcru.word3 << " 0x" << halfcru.word4 << " 0x" << halfcru.word57[0] << " 0x" << halfcru.word57[1] << " 0x" << halfcru.word57[2]<< std:: endl; 
    return stream;
}

} // namespace trd
} // namespace o2

#endif
