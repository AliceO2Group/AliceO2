// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RawDataFormat.h
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  TOF raw data format

#ifndef O2_TOF_RAWDATAFORMAT
#define O2_TOF_RAWDATAFORMAT

#include <stdint.h>

namespace o2
{
namespace tof
{
namespace raw
{

/** DRM data **/

struct DRMCommonHeader_t {
  uint32_t payload : 28;
  uint32_t wordType : 4;
};

struct DRMOrbitHeader_t {
  uint32_t orbit : 32;
};

struct DRMGlobalHeader_t {
  uint32_t slotID : 4;
  uint32_t eventWords : 17;
  uint32_t drmID : 7;
  uint32_t wordType : 4;
};

struct DRMStatusHeader1_t {
  uint32_t slotID : 4;
  uint32_t participatingSlotID : 11;
  uint32_t cBit : 1;
  uint32_t versID : 5;
  uint32_t drmHSize : 4;
  uint32_t undefined : 3;
  uint32_t wordType : 4;
};

struct DRMStatusHeader2_t {
  uint32_t slotID : 4;
  uint32_t slotEnableMask : 11;
  uint32_t mustBeZero : 1;
  uint32_t faultID : 11;
  uint32_t rtoBit : 1;
  uint32_t wordType : 4;
};

struct DRMStatusHeader3_t {
  uint32_t slotID : 4;
  uint32_t l0BunchID : 12;
  uint32_t runTimeInfo : 12; // check
  uint32_t wordType : 4;
};

struct DRMStatusHeader4_t {
  uint32_t slotID : 4;
  uint32_t temperature : 10;
  uint32_t mustBeZero1 : 1;
  uint32_t ackBit : 1;
  uint32_t sensAD : 3;
  uint32_t mustBeZero2 : 1;
  uint32_t undefined : 8;
  uint32_t wordType : 4;
};

struct DRMStatusHeader5_t {
  uint32_t unknown : 32;
};

struct DRMGlobalTrailer_t {
  uint32_t slotID : 4;
  uint32_t localEventCounter : 12;
  uint32_t undefined : 12;
  uint32_t wordType : 4;
};

/** TRM data **/

struct TRMGlobalHeader_t {
  uint32_t slotID : 4;
  uint32_t eventWords : 13;
  uint32_t eventNumber : 10;
  uint32_t eBit : 1;
  uint32_t wordType : 4;
};

struct TRMGlobalTrailer_t {
  uint32_t mustBeThree : 2;
  uint32_t eventCRC : 12;
  uint32_t temp : 8;
  uint32_t sendAd : 3;
  uint32_t chain : 1;
  uint32_t tsBit : 1;
  uint32_t lBit : 1;
  uint32_t wordType : 4;
};

/** TRM-chain data **/

struct TRMChainHeader_t {
  uint32_t slotID : 4;
  uint32_t bunchID : 12;
  uint32_t pb24Temp : 8;
  uint32_t pb24ID : 3;
  uint32_t tsBit : 1;
  uint32_t wordType : 4;
};

struct TRMChainTrailer_t {
  uint32_t status : 4;
  uint32_t mustBeZero : 12;
  uint32_t eventCounter : 12;
  uint32_t wordType : 4;
};

/** TDC hit **/

struct TDCPackedHit_t {
  uint32_t hitTime : 13;
  uint32_t totWidth : 8;
  uint32_t chan : 3;
  uint32_t tdcID : 4;
  uint32_t eBit : 1;
  uint32_t psBits : 2;
  uint32_t mustBeOne : 1;
};

struct TDCUnpackedHit_t {
  uint32_t hitTime : 21;  // leading or trailing edge measurement
  uint32_t chan : 3;      // TDC channel number
  uint32_t tdcID : 4;     // TDC ID
  uint32_t eBit : 1;      // E bit
  uint32_t psBits : 2;    // PS bits
  uint32_t mustBeOne : 1; // must-be-one bit
};

/** union **/

union Union_t {
  uint32_t data;
  DRMCommonHeader_t drmCommonHeader;
  DRMOrbitHeader_t drmOrbitHeader;
  DRMGlobalHeader_t drmGlobalHeader;
  DRMStatusHeader1_t drmStatusHeader1;
  DRMStatusHeader2_t drmStatusHeader2;
  DRMStatusHeader3_t drmStatusHeader3;
  DRMStatusHeader4_t drmStatusHeader4;
  DRMStatusHeader5_t drmStatusHeader5;
  DRMGlobalTrailer_t drmGlobalTrailer;
  TRMGlobalHeader_t trmGlobalHeader;
  TRMGlobalTrailer_t trmGlobalTrailer;
  TRMChainHeader_t trmChainHeader;
  TRMChainTrailer_t trmChainTrailer;
  TDCPackedHit_t tdcPackedHit;
  TDCUnpackedHit_t tdcUnpackedHit;
};

} // namespace raw
} // namespace tof
} // namespace o2

#define IS_DRM_COMMON_HEADER(x) ((x & 0xF0000000) == 0x40000000)
#define IS_DRM_GLOBAL_HEADER(x) ((x & 0xF000000F) == 0x40000001)
#define IS_DRM_GLOBAL_TRAILER(x) ((x & 0xF000000F) == 0x50000001)
#define IS_LTM_GLOBAL_HEADER(x) ((x & 0xF000000F) == 0x40000002)
#define IS_LTM_GLOBAL_TRAILER(x) ((x & 0xF000000F) == 0x50000002)
#define IS_TRM_GLOBAL_HEADER(x) ((x & 0xF0000000) == 0x40000000)
#define IS_TRM_GLOBAL_TRAILER(x) ((x & 0xF0000003) == 0x50000003)
#define IS_TRM_CHAINA_HEADER(x) ((x & 0xF0000000) == 0x00000000)
#define IS_TRM_CHAINA_TRAILER(x) ((x & 0xF0000000) == 0x10000000)
#define IS_TRM_CHAINB_HEADER(x) ((x & 0xF0000000) == 0x20000000)
#define IS_TRM_CHAINB_TRAILER(x) ((x & 0xF0000000) == 0x30000000)
#define IS_TDC_ERROR(x) ((x & 0xF0000000) == 0x60000000)
#define IS_FILLER(x) ((x & 0xFFFFFFFF) == 0x70000000)
#define IS_TDC_HIT(x) ((x & 0x80000000) == 0x80000000)

// DRM getters
#define GET_DRMGLOBALHEADER_DRMID(x) ((x & 0x0FE00000) >> 21)
#define GET_DRMSTATUSHEADER1_PARTICIPATINGSLOTID(x) ((x & 0x00007FF0) >> 4)
#define GET_DRMSTATUSHEADER1_CBIT(x) ((x & 0x00008000) >> 15)
#define GET_DRMSTATUSHEADER2_SLOTENABLEMASK(x) ((x & 0x00007FF0) >> 4)
#define GET_DRMSTATUSHEADER2_FAULTID(x) ((x & 0x07FF0000) >> 16)
#define GET_DRMSTATUSHEADER2_RTOBIT(x) ((x & 0x08000000) >> 27)
#define GET_DRMSTATUSHEADER3_L0BCID(x) ((x & 0x0000FFF0) >> 4)
#define GET_DRMGLOBALTRAILER_LOCALEVENTCOUNTER(x) ((x & 0x0000FFF0) >> 4)

// TRM getter
#define GET_TRMGLOBALHEADER_SLOTID(x) ((x & 0x0000000F))
#define GET_TRMGLOBALHEADER_EVENTNUMBER(x) ((x & 0x07FE0000) >> 17)
#define GET_TRM_EVENTWORDS(x) ((x & 0x0001FFF0) >> 4)
#define GET_TRMGLOBALHEADER_EBIT(x) ((x & 0x08000000) >> 27)

// TRM Chain getters
#define GET_TRMCHAINHEADER_SLOTID(x) ((x & 0x0000000F))
#define GET_TRMCHAINHEADER_BUNCHID(x) ((x & 0x0000FFF0) >> 4)
#define GET_TRMCHAINTRAILER_EVENTCOUNTER(x) ((x & 0x0FFF0000) >> 16)
#define GET_TRMCHAINTRAILER_STATUS(x) ((x & 0x0000000F))

// TDC getters
#define GET_TDCHIT_HITTIME(x) ((x & 0x001FFFFF))
#define GET_TDCHIT_CHAN(x) ((x & 0x00E00000) >> 21)
#define GET_TDCHIT_TDCID(x) ((x & 0x0F000000) >> 24)
#define GET_TDCHIT_EBIT(x) ((x & 0x10000000) >> 28)
#define GET_TDCHIT_PSBITS(x) ((x & 0x60000000) >> 29)

#endif /** O2_TOF_RAWDATAFORMAT **/
