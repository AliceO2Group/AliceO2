// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//#include "TRDBase/TRDGeometryBase.h"
//#include "DetectorsCommonDataFormats/DetMatrixCache.h"
//#include "DetectorsCommonDataFormats/DetID.h"

#ifndef O2_TRD_COMPRESSEDHEADER_H
#define O2_TRD_COMPRESSEDHEADER_H

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TRD Compressed Header                                                  //
// struct to hold the header for the raw compressed data of tracklet64
// when compression happens on the flp
// Authors                                                                //
//  Sean Murray (murrays@cern.ch)                                         //
//
////////////////////////////////////////////////////////////////////////////
#include "Rtypes.h" // for ClassDef
#include "fairlogger/Logger.h"

namespace o2
{
namespace trd
{
/*      |63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|
        -------------------------------------------------------------------------------------------------
Word 0  |   Format  |              time since frame start                                               |
        -------------------------------------------------------------------------------------------------
        |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
        -------------------------------------------------------------------------------------------------
Word 0  |                                    length of data block                                       |
        -------------------------------------------------------------------------------------------------
This is chosen to be a 64 bit word as this is essentially a fake tracklet64 posing as a header.
This then explains the sizes.
Format is 4 bits, we only need 3 formats for now.
*/
struct CompressedRawHeader {
  union {
    uint64_t word0;
    struct {
      uint64_t size : 32;      // size of curent block, including this header in 64bit words.
      uint64_t eventtime : 28; // time from beginning of the time frame
      uint8_t format : 4;      // format of data 1=tracklets,2=digits, 3=config.
    } __attribute__((__packed__));
    uint64_t word1; // stores the information that will end up in the triggerrecord.
    struct {
      uint16_t bc : 16;      // bunch crossing coming from rdh
      uint32_t orbit : 32;   // orbit of lhc
      uint16_t padding : 16; // 0xeeee
    } __attribute__((__packed__));
  };
};
//This is simply 64bits of e to mark the end and enable some form of error checking.
//this appears at the end of tracklets.
//i.e. start pointer + header.size -1  == Compressed Trailer position.
struct CompressedRawTrackletDigitSeperator {
  union {
    uint64_t word; //0xeeeeeexxxxeeeeeeLL where xxxx is the number of digits to follow, max digits on a link is 15*8*16*21 (links*rob*mcm*adc)
    //TODO I think is off by a factor of 2 but does not matter, the first 'x' is padded in anycase, so would not save in the hex definition above.
    struct {
      uint32_t pad2 : 24;       // padding e as a marker
      uint16_t digitcount : 16; // count of digits to come
      uint32_t pad1 : 24;       // padding e as marker
    } __attribute__((__packed__));
  };
};

/*      |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
        -------------------------------------------------------------------------------------------------
Word    |   detector               |  rob   | mcm       |    ????????????                               |
        -------------------------------------------------------------------------------------------------
time is supplied by the compressedrawheader of the tracklets which is always there.
detector [0-540] 9 bits , rob [0-8] 3 bits , mcm [0-16] 4
= 15 bits ... what to do with the other 17 bits ?
Word of caution, tracklets word size is 64 bit digit word size is 32.
So the raw data has 2 *different* word sizes!
TODO ignoring config data of course.
digits are padded at end with a digit end marker (32 bits of 0xe) and then padded to 64 bits.
i.e. possibly 64 bits of 0xe
TODO in real data are the digits sent in zero suppresed?
*/

struct CompressedRawDigitHeader {
  union {
    uint64_t word;
    struct {
      uint32_t padding : 16; // padding e as a marker
      uint16_t mcm : 4;      // count of digits to come
      uint16_t rob : 3;      // count of digits to come
      uint32_t dector : 9;   // detector number
    } __attribute__((__packed__));
  };
};

struct CompressedRawDigitEndMarker {
  uint32_t word; // 0xeeeeeeee can be doubled up to pad to 64bit wide data.
};

//For now we ignore config data.
//TODO add config data ....
//
} //namespace trd
} //namespace o2
#endif
