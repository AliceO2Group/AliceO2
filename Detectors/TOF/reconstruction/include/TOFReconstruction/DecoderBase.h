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

/// @file   Decoder.h
/// @author Roberto Preghenella
/// @since  2020-02-24
/// @brief  TOF compressed data decoder

#ifndef O2_TOF_DECODERBASE
#define O2_TOF_DECODERBASE

#include <fstream>
#include <string>
#include <cstdint>
#include <vector>
#include "Headers/RAWDataHeader.h"
#include "DataFormatsTOF/CompressedDataFormat.h"

namespace o2
{
namespace tof
{
namespace compressed
{

template <typename RDH>
class DecoderBaseT
{

 public:
  DecoderBaseT() = default;
  virtual ~DecoderBaseT() = default;

  inline bool run()
  {
    rewind();
    if (mDecoderCONET) {
      mDecoderPointerMax = reinterpret_cast<const uint32_t*>(mDecoderBuffer + mDecoderBufferSize);
      while (mDecoderPointer < mDecoderPointerMax) {
        if (processDRM()) {
          return false;
        }
      }
      return false;
    }
    while (!processHBF()) {
      ;
    }
    return false;
  };

  inline void rewind()
  {
    decoderRewind();
  };

  void setDecoderVerbose(bool val) { mDecoderVerbose = val; };
  void setDecoderBuffer(const char* val) { mDecoderBuffer = val; };
  void setDecoderBufferSize(long val) { mDecoderBufferSize = val; };
  void setDecoderCONET(bool val) { mDecoderCONET = val; };

 private:
  /** handlers **/

  virtual void rdhHandler(const RDH* rdh) = 0;
  virtual void headerHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit) = 0;

  virtual void frameHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
                            const FrameHeader_t* frameHeader, const PackedHit_t* packedHits) = 0;

  virtual void trailerHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
                              const CrateTrailer_t* crateTrailer, const Diagnostic_t* diagnostics,
                              const Error_t* errors) = 0;

  bool processHBF();
  bool processDRM();

  /** decoder private functions and data members **/
  inline void decoderRewind() { mDecoderPointer = reinterpret_cast<const uint32_t*>(mDecoderBuffer); };

  const char* mDecoderBuffer = nullptr;
  long mDecoderBufferSize;
  const uint32_t* mDecoderPointer = nullptr;
  const uint32_t* mDecoderPointerMax = nullptr;
  const uint32_t* mDecoderPointerNext = nullptr;
  const RDH* mDecoderRDH;
  bool mDecoderVerbose = false;
  bool mDecoderError = false;
  bool mDecoderFatal = false;
  bool mDecoderCONET = false;
  char mDecoderSaveBuffer[1048576];
  uint32_t mDecoderSaveBufferDataSize = 0;
  uint32_t mDecoderSaveBufferDataLeft = 0;
};

typedef DecoderBaseT<o2::header::RAWDataHeaderV4> DecoderBaseV4;
typedef DecoderBaseT<o2::header::RAWDataHeaderV6> DecoderBaseV6;
typedef DecoderBaseT<o2::header::RAWDataHeaderV7> DecoderBaseV7;
using DecoderBase = DecoderBaseT<o2::header::RAWDataHeader>;

} // namespace compressed
} // namespace tof
} // namespace o2

#endif /** O2_TOF_DECODERBASE **/
