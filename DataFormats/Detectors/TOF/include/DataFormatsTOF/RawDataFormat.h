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
/// @author Pietro Antonioli
/// @since  2019-12-18
/// @brief  TOF raw data format

#ifndef O2_TOF_RAWDATAFORMAT
#define O2_TOF_RAWDATAFORMAT

#include <cstdint>

#ifndef O2_TOF_RAWDATAFORMAT_NONAMESPACE
namespace o2
{
namespace tof
{
namespace raw
{
#endif

/*

   ALICE TOF Raw Data description: master file

   Header file to describe TOF event fragment: defines, macros and data structures


   @P. Antonioli / INFN-Bologna
   March 2006

   Dec. 2019 moved old CDH to new TDH, updated format for RUN3: new DRM2, new TRM format

*/

/* GEO Ad assigned to TOF modules */
#define DRM_GEOAD 1
#define LTM_GEOAD 2
#define TRM_GEOAD_MIN 3
#define TRM_GEOAD_MAX 12

/* Symbols and macros valid for all slots */
#define TOF_HEADER 4
#define TOF_TRAILER 5
#define TOF_FILLER 7
#define FILLER_WORD (TOF_FILLER << 28)
#define TOF_GETGEO(x) (x & 0xF)
#define TOF_GETDATAID(x) ((x >> 28) & 0xF)

/* TOF Data Header (former CDH)
        |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
        -------------------------------------------------------------------------------------------------
Word 0  |   0100    |    00                       |        Event Length (Bytes)                         |
        -------------------------------------------------------------------------------------------------
Word 1  |                                     LHC Orbit                                                 |
        -------------------------------------------------------------------------------------------------
*/
// TDH - TOF Data Header
typedef struct {
  uint32_t bytePayload : 18;
  uint32_t mbz : 10;
  uint32_t dataId : 4;
} __attribute__((__packed__)) TOFDataHeader_t;

typedef struct {
  uint32_t orbit : 32;
} __attribute__((__packed__)) TOFOrbit_t;

typedef struct {
  TOFDataHeader_t head;
  TOFOrbit_t orbit;
} __attribute__((__packed__)) TDH_t;

#define TDH_SIZE sizeof(TDH_t) // size is in bytes length is in words
#define TDH_LENGTH TDH_SIZE / 4

#define TDH_HEADER(d) TOF_GETDATAID(d)
#define TDH_PAYLOAD(d) (d & 0x3FFFF)
#define TDH_WORDS(d) TDH_PAYLOAD(d) / 4
#define TDH_ORBIT(d) (d & 0xFFFFFFFF) //32 bit

/* DRM headers & trailer 

        |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
        -------------------------------------------------------------------------------------------------
Word 0  |   0100    |   Drm Id           |                 Event Length (Words)             |  0001     |
        -------------------------------------------------------------------------------------------------
Word 1  |   0100    | 0   | DRMHWords | DRMHVersion  | CLK |  Slot Participating Mask       |  0001     |
        -------------------------------------------------------------------------------------------------
Word 2  |   0100    |TO|  Slot Fault Mask               |0 |  Slot Enable Mask              |  0001     |
        -------------------------------------------------------------------------------------------------
Word 3  |   0100    |   Bunch Crossing (Local)          |    Bunch Crossing (GBTx)          |  0001     |
        -------------------------------------------------------------------------------------------------
Word 4  |   0100    |        00       |  Temp Ad  | 00  |   Temp Value                      |  0001     |
        -------------------------------------------------------------------------------------------------
Word 5  |   0100    |        00       |IR|                   Event CRC                      |  0001     |
        -------------------------------------------------------------------------------------------------
Trailer |   0101    |        000                        |    Event Counter (Local)          |  0001     |
        -------------------------------------------------------------------------------------------------
*/
// DRMH - DRM Data Header and Trailer
//----------------------------------- Word 0
typedef struct {
  uint32_t slotId : 4;
  uint32_t eventWords : 17; // meglio portarlo a 16
  uint32_t drmId : 7;       // 0-71== 0x0-0x47 (Bit 8 > 0x100 announnce DRM generated payload, da cambiare in 0x80 e renderlo a 8 bit)
  uint32_t dataId : 4;
} __attribute__((__packed__)) DRMDataHeader_t;

//----------------------------------- Word 1
typedef struct {
  uint32_t slotId : 4;
  uint32_t partSlotMask : 11; // meglio mettere un mbz di 1 cosi' rispetta i byte boundaries
  uint32_t clockStatus : 2;
  uint32_t drmhVersion : 5; // currently set to 0x11 (Bit 4 identifies RUN3/4 data format), puo' essere diminuito a 6
  uint32_t drmHSize : 4;    // it doesn't count previous word, so currently it is 5
  uint32_t mbz : 2;
  uint32_t dataId : 4;
} __attribute__((__packed__)) DRMHeadW1_t;
//----------------------------------- Word 2
typedef struct {
  uint32_t slotId : 4;
  uint32_t enaSlotMask : 11;
  uint32_t mbz : 1;
  uint32_t faultSlotMask : 11;
  uint32_t readoutTimeOut : 1;
  uint32_t dataId : 4;
} __attribute__((__packed__)) DRMHeadW2_t;
//----------------------------------- Word 3
typedef struct {
  uint32_t slotId : 4;
  uint32_t gbtBunchCnt : 12;
  uint32_t locBunchCnt : 12;
  uint32_t dataId : 4;
} __attribute__((__packed__)) DRMHeadW3_t;
//----------------------------------- Word 4
typedef struct {
  uint32_t slotId : 4;
  uint32_t tempValue : 10;
  uint32_t mbza : 2;
  uint32_t tempAddress : 4;
  uint32_t mbzb : 8;
  uint32_t dataId : 4;
} __attribute__((__packed__)) DRMHeadW4_t;
//----------------------------------- Word 5
typedef struct {
  uint32_t slotId : 4;
  uint32_t eventCRC : 16;
  uint32_t irq : 1;
  uint32_t mbz : 7;
  uint32_t dataId : 4;
} __attribute__((__packed__)) DRMHeadW5_t;

// full DRMH (header + 5 words)
typedef struct {
  DRMDataHeader_t head;
  DRMHeadW1_t w1;
  DRMHeadW2_t w2;
  DRMHeadW3_t w3;
  DRMHeadW4_t w4;
  DRMHeadW5_t w5;
} __attribute__((__packed__)) DRMh_t;

typedef struct {
  uint32_t slotId : 4;
  uint32_t locEvCnt : 12;
  uint32_t mbz : 12; // qui si potrebbero mettere 8 bit di trigger received nell'orbita finora
  uint32_t dataId : 4;
} __attribute__((__packed__)) DRMDataTrailer_t;

#define DRMH_SIZE sizeof(DRMh_t)
#define DRMH_LENGTH DRMH_SIZE / 4
#define DRM_HEAD_NW DRMH_LENGTH

// Word 0
#define DRM_DRMID(a) ((a & 0x007E00000) >> 21)  //was FE    ///CHECK
#define DRM_EVWORDS(a) ((a & 0x0001FFFF0) >> 4) ///CHECK
// Word 1
#define DRM_SLOTID(a) ((a & 0x00007FF0) >> 4)
#define DRM_CLKFLG(a) ((a & 0x00018000) >> 15) //was 10000>>15
#define DRM_VERSID(a) ((a & 0x003E0000) >> 17) //waa 1E0000>>16
#define DRM_HSIZE(a) ((a & 0x03C00000) >> 22)  //was 3E0000>>21
// Word 2
#define DRM_ENABLEID(a) ((a & 0x00007FF0) >> 4)
#define DRM_FAULTID(a) ((a & 0x07FF0000) >> 16)
#define DRM_RTMO(a) ((a & 0x08000000) >> 27)
// Word 3
#define DRM_BCGBT(a) ((a & 0x0000FFF0) >> 4)
#define DRM_BCLOC(a) ((a & 0x0FFF0000) >> 16)
// Word 4
#define DRM_TEMP(a) ((a & 0x00003FF0) >> 4)
#define DRM_SENSAD(a) ((a & 0x00070000) >> 18)
// Word 5
#define DRM_EVCRC(a) ((a & 0x000FFFF0) >> 4)
// Trailer
#define DRM_LOCEVCNT(a) ((a & 0x0000FFF0) >> 4)

/* TRM headers & trailers */
#define TRM_HEADER TOF_HEADER
#define TRM_TRAILER TOF_TRAILER
#define CHAIN_0_HEADER 0
#define CHAIN_0_TRAILER 1
#define CHAIN_1_HEADER 2
#define CHAIN_1_TRAILER 3
#define HIT_LEADING 0xA
#define HIT_TRAILING 0xC
#define REPORT_ERROR 6
#define DEBERR REPORT_ERROR

#define TRM_WORDID(a) TOF_GETDATAID(a) // legacy

/* TRM Global Header                 
  |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
  -------------------------------------------------------------------------------------------------
  |   0100    | E|         EVENT NUMBER (10)   |             EVENT WORDS (13)         |  SLOT ID  |
                |
                |__ Empty event
*/
// TRM - TRM Data Global and Chain Header and Trailer
typedef struct {
  uint32_t slotId : 4;
  uint32_t eventWords : 13;
  uint32_t eventCnt : 10;
  uint32_t emptyBit : 1;
  uint32_t dataId : 4;
} __attribute__((__packed__)) TRMDataHeader_t;
#define TRM_EVCNT_GH(a) ((a & 0x07FE0000) >> 17)
#define TRM_EVWORDS(a) ((a & 0x0001FFF0) >> 4)

/* TRM Chain Header
  |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
  -------------------------------------------------------------------------------------------------
  | 0000/0010 |        RESERVED                   |             BUNCH ID              |  SLOT ID  |
*/
typedef struct {
  uint32_t slotId : 4;
  uint32_t bunchCnt : 12;
  uint32_t mbz : 12;
  uint32_t dataId : 4; // bit 29 flag the chain
} __attribute__((__packed__)) TRMChainHeader_t;
#define TRM_BUNCHID(a) ((a & 0x0000FFF0) >> 4)

/* TRM Chain Trailer
  |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
  -------------------------------------------------------------------------------------------------
  | 0001/0011 |        EVENT COUNTER              |           CHAIN_EVNT_WORD (12)    |  STATUS   |
*/
typedef struct {
  uint32_t status : 4;
  uint32_t mbz : 12;
  uint32_t eventCnt : 12;
  uint32_t dataId : 4; // bit 29 flag the chain
} __attribute__((__packed__)) TRMChainTrailer_t;
#define TRM_EVCNT_CT(a) ((a & 0x0FFF0000) >> 16)
#define TRM_CHAINSTAT(a) (a & 0xF)

/* TRM Global Trailer  
  |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
  -------------------------------------------------------------------------------------------------
  |   0101    | L|TS|CN| SENS AD|        TEMP (8)       |     EVENT CRC (12)                | X| X|
                |  |  |
                |  |  |__ Chain
                |  |__ Temp Status bit (1=OK,0=not valid)
                |__ Lut error
*/
typedef struct {
  uint32_t trailerMark : 2;
  uint32_t eventCRC : 12;
  uint32_t tempValue : 8;
  uint32_t tempAddress : 3;
  uint32_t tempChain : 1;
  uint32_t tempAck : 1;
  uint32_t lutErrorBit : 1;
  uint32_t dataId : 4;
} __attribute__((__packed__)) TRMDataTrailer_t;

#define TRM_LUTERRBIT(a) ((a & 0x08000000) >> 27)
#define TRM_PB24TEMP(a) ((a & 0x003FC000) >> 14)
#define TRM_PB24ID(a) ((a & 0x01C00000) >> 22)
#define TRM_PB24CHAIN(a) ((a & 0x02000000) >> 25)
#define TRM_PB24ACK(a) ((a & 0x04000000) >> 26)
#define TRM_EVCRC2(a) ((a & 0x00003FFC) >> 2)
#define TRM_TERM(a) (a & 0x3)

//#define TRM_EVCNT2(a)    ((a & 0x07FE0000)>>17)
//#define TRM_EVCRC(a)     ((a & 0x0000FFF0)>>4)

// TDC Hit Decoding
/* 
  |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
  -------------------------------------------------------------------------------------------------
  | Data Id   | TDC Id    | Ch Id  | TIME                                                         |       
 
   Note on dataId: all values > 5 here:  
   0110 HPTDC error or HPTDC test word (or global error if TDCId=0xF)
   1010 Hit leading
   1100 Hit trailing
   if bit 28 is on, both copies of LUT corrupted, wrong compensation applied
*/
typedef struct {
  uint32_t time : 21;
  uint32_t chanId : 3;
  uint32_t tdcId : 4;
  uint32_t dataId : 4;
} __attribute__((__packed__)) TRMDataHit_t;

#define TRM_TIME(a) (a & 0x1FFFFF)
#define TRM_CHANID(a) ((a >> 21) & 0x7)
#define TRM_TDCID(a) ((a >> 24) & 0xF)

// LTM
// LTM - LTM Data Header and Trailer
typedef struct {
  uint32_t slotId : 4;
  uint32_t eventWords : 13;
  uint32_t cycloneErr : 1;
  uint32_t fault : 6;
  uint32_t mbz0 : 4;
  uint32_t dataId : 4;
} __attribute__((__packed__)) LTMDataHeader_t;
#define LTM_HEAD_TAG(x) TOF_GETDATAID(x)
#define LTM_HEAD_FAULTFLAG(x) (((x) >> 18) & 0x1)
#define LTM_HEAD_CYCSTATUS(x) (((x) >> 17) & 0x1)
#define LTM_EVENTSIZE(x) (((x) >> 4) & 0x1FFF) // actual event size
#define LTM_HEAD_EVENTSIZE(x) (((x) >> 4) & 0x1FFF)
#define LTM_HEAD_GEOADDR(x) TOF_GETGEO(x)
#define LTM_HEADER TOF_HEADER

typedef struct {
  uint32_t slotId : 4;
  uint32_t eventCRC : 12;
  uint32_t eventCnt : 12;
  uint32_t dataId : 4;
} __attribute__((__packed__)) LTMDataTrailer_t;
#define LTM_TAIL_TAG(x) TOF_GETDATAID(x)
#define LTM_TAIL_EVENTNUM(x) (((x) >> 16) & 0xFFF)
#define LTM_TAIL_EVENTCRC(x) (((x) >> 4) & 0xFFF)
#define LTM_TAIL_GEOADDR(x) TOF_GETGEO(x)
#define LTM_TRAILER TOF_TRAILER

typedef struct {
  uint32_t pdl0 : 8;
  uint32_t pdl1 : 8;
  uint32_t pdl2 : 8;
  uint32_t pdl3 : 8;
} __attribute__((__packed__)) LTMPdlWord_t;

typedef struct {
  uint32_t adc0 : 10;
  uint32_t adc1 : 10;
  uint32_t adc2 : 10;
  uint32_t mbz : 2;
} __attribute__((__packed__)) LTMAdcWord_t;

typedef struct {
  LTMDataHeader_t head;
  LTMPdlWord_t pdlData[12];
  //  48 PDL values        12 words
  LTMAdcWord_t adcData[36];
  //  16 Low Voltages       5 words  + 10 bit
  //  16 Thresholds         5 words  + 10 bit
  //   8 FEAC GND           2 words  + 10 bit
  //   8 FEAC Temp          2 words  + 10 bit
  //  12 LTM  Temp:         4 words
  //  48 OR trigger rates: 16 words
  //                       34 words + 2 words = 36 words
  LTMDataTrailer_t trailer;
} __attribute__((__packed__)) LTMPackedEvent_t;
#define LTM_EVSIZE sizeof(ltmPackedEvent_t) // fixed expected event size
#define LTM_PDL_FIELD(x, n) (((x) >> ((n)*8)) & 0xFF)
#define LTM_V_FIELD(x, n) (((x) >> ((n)*10)) & 0x3FF)
#define LTM_T_FIELD(x, n) (((x) >> ((n)*10)) & 0x3FF)
#define LTM_OR_FIELD(x, n) (((x) >> ((n)*10)) & 0x3FF)

#define ltmint unsigned int //si puo' portarlo a uint32_t rendendo la struttura piu' piccola, da verificare se non ci sono overflow
typedef struct {
  ltmint TagHead;
  ltmint FaultFlag;
  ltmint CycStatus;
  ltmint EventSize;
  ltmint HeadGeo;
  ltmint PdlDelay[48];
  ltmint Vlv[16];
  ltmint Vth[16];
  ltmint GndFeac[8];
  ltmint FeacTemp[8];
  ltmint LocalTemp[12];
  ltmint OrRate[48];
  ltmint EventNum;
  ltmint EventCrc;
  ltmint TailGeo;
  ltmint TagTail;
} LTMEvent_t;

/** union **/

typedef union {
  uint32_t data;
  TOFDataHeader_t tofDataHeader;
  TOFOrbit_t tofOrbit;
  DRMDataHeader_t drmDataHeader;
  DRMHeadW1_t drmHeadW1;
  DRMHeadW2_t drmHeadW2;
  DRMHeadW3_t drmHeadW3;
  DRMHeadW4_t drmHeadW4;
  DRMHeadW5_t drmHeadW5;
  DRMDataTrailer_t drmDataTrailer;
  TRMDataHeader_t trmDataHeader;
  TRMDataTrailer_t trmDataTrailer;
  TRMChainHeader_t trmChainHeader;
  TRMChainTrailer_t trmChainTrailer;
  TRMDataHit_t trmDataHit;
} Union_t;

#ifndef O2_TOF_RAWDATAFORMAT_NONAMESPACE
} // namespace raw
} // namespace tof
} // namespace o2
#endif

#endif /** O2_TOF_RAWDATAFORMAT **/
