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
struct CompressedHeader {
  union {
    uint64_t word;
    struct {
      uint8_t format : 4;      // format of data 1=tracklets,2=digits, 3=config.
      uint64_t eventtime : 28; // time from beginning of the time frame
      uint64_t size : 32;      // size of curent block, including this header in 64bit words.
    } __attribute__((__packed__));
  };
};

} //namespace trd
} //namespace o2
#endif
