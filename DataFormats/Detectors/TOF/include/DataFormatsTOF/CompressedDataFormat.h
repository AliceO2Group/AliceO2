// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CmpDataFormat.h
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  TOF compressed data format

#ifndef O2_TOF_COMPRESSEDDATAFORMAT
#define O2_TOF_COMPRESSEDDATAFORMAT

#include <cstdint>

namespace o2
{
namespace tof
{
namespace compressed
{

struct Word_t {
  uint32_t undefined : 31;
  uint32_t wordType : 1;
};

struct CrateHeader_t {
  uint32_t bunchID : 12;
  uint32_t slotPartMask : 11;
  uint32_t undefined : 1;
  uint32_t drmID : 7;
  uint32_t mustBeOne : 1;
  static const uint32_t slotEnableMask = 0x0; // deprecated
};

struct CrateOrbit_t {
  uint32_t orbitID : 32;
};

struct FrameHeader_t {
  uint32_t numberOfHits : 16;
  uint32_t frameID : 8;
  uint32_t trmID : 4;
  uint32_t deltaBC : 3;
  uint32_t mustBeZero : 1;
};

struct PackedHit_t {
  uint32_t tot : 11;
  uint32_t time : 13;
  uint32_t channel : 3;
  uint32_t tdcID : 4;
  uint32_t chain : 1;
};

struct CrateTrailer_t {
  uint32_t numberOfDiagnostics : 4;
  uint32_t eventCounter : 12;
  uint32_t numberOfErrors : 9;
  uint32_t undefined : 15;
  uint32_t mustBeOne : 1;
};

struct Diagnostic_t {
  uint32_t slotID : 4;
  uint32_t faultBits : 28;
};

struct Error_t {
  uint32_t errorFlags : 15;
  uint32_t undefined : 4;
  uint32_t slotID : 4;
  uint32_t chain : 1;
  uint32_t tdcID : 4;
  uint32_t mustBeSix : 4;
};

/** union **/

union Union_t {
  uint32_t data;
  Word_t word;
  CrateHeader_t crateHeader;
  CrateOrbit_t crateOrbit;
  FrameHeader_t frameHeader;
  PackedHit_t packedHit;
  CrateTrailer_t crateTrailer;
};

} // namespace compressed

namespace diagnostic
{

/** DRM diagnostic bits, 12 bits [4-15] **/
enum EDRMDiagnostic_t {
  DRM_HEADER_MISSING = 1 << 4, // start from BIT(4)
  DRM_TRAILER_MISSING = 1 << 5,
  DRM_FEEID_MISMATCH = 1 << 6,
  DRM_ORBIT_MISMATCH = 1 << 7,
  DRM_CRC_MISMATCH = 1 << 8,
  DRM_ENAPARTMASK_DIFFER = 1 << 9,
  DRM_CLOCKSTATUS_WRONG = 1 << 10,
  DRM_FAULTSLOTMASK_NOTZERO = 1 << 11,
  DRM_READOUTTIMEOUT_NOTZERO = 1 << 12,
  DRM_EVENTWORDS_MISMATCH = 1 << 13,
  DRM_DIAGNOSTIC_SPARE1 = 1 << 14,
  DRM_DECODE_ERROR = 1 << 15,
  DRM_MAXDIAGNOSTIC_BIT = 1 << 16 // end before BIT(16)
};

static const char* DRMDiagnosticName[32] = {
  "DRM_HAS_DATA",
  "",
  "",
  "",
  "DRM_HEADER_MISSING",
  "DRM_TRAILER_MISSING",
  "DRM_FEEID_MISMATCH",
  "DRM_ORBIT_MISMATCH",
  "DRM_CRC_MISMATCH",
  "DRM_ENAPARTMASK_DIFFER",
  "DRM_CLOCKSTATUS_WRONG",
  "DRM_FAULTSLOTMASK_NOTZERO",
  "DRM_READOUTTIMEOUT_NOTZERO",
  "DRM_EVENTWORDS_MISMATCH",
  "DRM_DIAGNOSTIC_SPARE1",
  "DRM_DECODE_ERROR"};

/** LTM diagnostic bits **/
enum ELTMDiagnostic_t {
  LTM_HEADER_MISSING = 1 << 4, // start from BIT(4)
  LTM_TRAILER_MISSING = 1 << 5,
  LTM_DIAGNOSTIC_SPARE1 = 1 << 6,
  LTM_HEADER_UNEXPECTED = 1 << 7,
  LTM_MAXDIAGNOSTIC_BIT = 1 << 16 // end before BIT(16)
};

static const char* LTMDiagnosticName[32] = {
  "LTM_HAS_DATA",
  "",
  "",
  "",
  "LTM_HEADER_MISSING",
  "LTM_TRAILER_MISSING",
  "LTM_DIAGNOSTIC_SPARE1",
  "LTM_HEADER_UNEXPECTED"};

/** TRM diagnostic bits, 12 bits [4-15] **/
enum ETRMDiagnostic_t {
  TRM_HEADER_MISSING = 1 << 4, // start from BIT(4)
  TRM_TRAILER_MISSING = 1 << 5,
  TRM_CRC_MISMATCH = 1 << 6,
  TRM_HEADER_UNEXPECTED = 1 << 7,
  TRM_EVENTCNT_MISMATCH = 1 << 8,
  TRM_EMPTYBIT_NOTZERO = 1 << 9,
  TRM_LBIT_NOTZERO = 1 << 10,
  TRM_FAULTSLOTBIT_NOTZERO = 1 << 11,
  TRM_EVENTWORDS_MISMATCH = 1 << 12,
  TRM_DIAGNOSTIC_SPARE1 = 1 << 13,
  TRM_DIAGNOSTIC_SPARE2 = 1 << 14,
  TRM_DECODE_ERROR = 1 << 15,
  TRM_DIAGNOSTIC_SPARE3 = TRM_DECODE_ERROR, // backward compatibility
  TRM_MAXDIAGNOSTIC_BIT = 1 << 16           // end before BIT(16)
};

/** TRM Chain diagnostic bits, 8 bits [16-23] chainA [24-31] chainB **/
enum ETRMChainDiagnostic_t {
  TRMCHAIN_HEADER_MISSING = 1 << 16, // start from BIT(14), BIT(24)
  TRMCHAIN_TRAILER_MISSING = 1 << 17,
  TRMCHAIN_STATUS_NOTZERO = 1 << 18,
  TRMCHAIN_EVENTCNT_MISMATCH = 1 << 19,
  TRMCHAIN_TDCERROR_DETECTED = 1 << 20,
  TRMCHAIN_BUNCHCNT_MISMATCH = 1 << 21,
  TRMCHAIN_DIAGNOSTIC_SPARE1 = 1 << 22,
  TRMCHAIN_DIAGNOSTIC_SPARE2 = 1 << 23,
  TRMCHAIN_MAXDIAGNOSTIC_BIT = 1 << 24 // end before BIT(23), BIT(32)
};

static const char* TRMDiagnosticName[32] = {
  "TRM_HAS_DATA",
  "",
  "",
  "",
  "TRM_HEADER_MISSING",
  "TRM_TRAILER_MISSING",
  "TRM_CRC_MISMATCH",
  "TRM_HEADER_UNEXPECTED",
  "TRM_EVENTCNT_MISMATCH",
  "TRM_EMPTYBIT_NOTZERO",
  "TRM_LBIT_NOTZERO",
  "TRM_FAULTSLOTBIT_NOTZERO",
  "TRM_EVENTWORDS_MISMATCH",
  "TRM_DIAGNOSTIC_SPARE1",
  "TRM_DIAGNOSTIC_SPARE2",
  "TRM_DECODE_ERROR",
  "TRM_CHAIN_A_HEADER_MISSING",
  "TRM_CHAIN_A_TRAILER_MISSING",
  "TRM_CHAIN_A_STATUS_NOTZERO",
  "TRM_CHAIN_A_EVENTCNT_MISMATCH",
  "TRM_CHAIN_A_TDCERROR_DETECTED",
  "TRM_CHAIN_A_BUNCHCNT_MISMATCH",
  "TRM_CHAIN_A_DIAGNOSTIC_SPARE1",
  "TRM_CHAIN_A_DIAGNOSTIC_SPARE2",
  "TRM_CHAIN_B_HEADER_MISSING",
  "TRM_CHAIN_B_TRAILER_MISSING",
  "TRM_CHAIN_B_STATUS_NOTZERO",
  "TRM_CHAIN_B_EVENTCNT_MISMATCH",
  "TRM_CHAIN_B_TDCERROR_DETECTED",
  "TRM_CHAIN_B_BUNCHCNT_MISMATCH",
  "TRM_CHAIN_B_DIAGNOSTIC_SPARE1",
  "TRM_CHAIN_B_DIAGNOSTIC_SPARE2"};

} // namespace diagnostic

} // namespace tof
} // namespace o2

#endif /** O2_TOF_CMPDATAFORMAT **/
