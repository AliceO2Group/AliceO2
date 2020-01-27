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
  uint32_t slotEnableMask : 11;
  uint32_t undefined : 1;
  uint32_t drmID : 7;
  uint32_t mustBeOne : 1;
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
  uint32_t undefined : 15;
  uint32_t mustBeOne : 1;
};

struct Diagnostic_t {
  uint32_t slotID : 4;
  uint32_t faultBits : 28;
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
} // namespace tof
} // namespace o2

#define DIAGNOSTIC_DRM_HEADER_MISSING 0x80000000
#define DIAGNOSTIC_DRM_TRAILER_MISSING 0x40000000
#define DIAGNOSTIC_DRM_CRC 0x20000000
#define DIAGNOSTIC_DRM_ENAPARTMASK_DIFFER 0x08000000
#define DIAGNOSTIC_DRM_CLOCKSTATUS_WRONG 0x04000000
#define DIAGNOSTIC_DRM_FAULTSLOTMASK_NOTZERO 0x02000000
#define DIAGNOSTIC_DRM_READOUTTIMEOUT_NOTZERO 0x01000000

#define DIAGNOSTIC_TRM_HEADER_MISSING 0x80000000
#define DIAGNOSTIC_TRM_TRAILER_MISSING 0x40000000
#define DIAGNOSTIC_TRM_CRC 0x20000000
#define DIAGNOSTIC_TRM_HEADER_UNEXPECTED 0x20000000
#define DIAGNOSTIC_TRM_EVENTCNT_MISMATCH 0x08000000
#define DIAGNOSTIC_TRM_EMPTYBIT_NOTZERO 0x06000000
#define DIAGNOSTIC_TRM_LBIT 0x02000000

#define DIAGNOSTIC_TRMCHAIN_HEADER_MISSING(x) (0x00080000 << (8 * x))
#define DIAGNOSTIC_TRMCHAIN_TRAILER_MISSING(x) (0x00040000 << (8 * x))
#define DIAGNOSTIC_TRMCHAIN_STATUS_NOTZERO(x) (0x00020000 << (8 * x))
#define DIAGNOSTIC_TRMCHAIN_EVENTCNT_MISMATCH(x) (0x00008000 << (8 * x))
#define DIAGNOSTIC_TRMCHAIN_TDCERROR_DETECTED(x) (0x00004000 << (8 * x))
#define DIAGNOSTIC_TRMCHAIN_BUNCHCNT_MISMATCH(x) (0x00002000 << (8 * x))

#endif /** O2_TOF_CMPDATAFORMAT **/
