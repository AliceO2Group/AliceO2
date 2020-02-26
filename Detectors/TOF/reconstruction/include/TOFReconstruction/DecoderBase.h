// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

class DecoderBase
{

 public:
  DecoderBase() = default;
  ~DecoderBase() = default;

  inline bool run()
  {
    rewind();
    while (!processHBF())
      ;
    return false;
  };

  inline void rewind()
  {
    decoderRewind();
  };

  void setDecoderVerbose(bool val) { mDecoderVerbose = val; };
  void setDecoderBuffer(char* val) { mDecoderBuffer = val; };
  void setDecoderBufferSize(long val) { mDecoderBufferSize = val; };

 private:
  /** handlers **/

  virtual void rdhHandler(const o2::header::RAWDataHeader* rdh){};
  virtual void headerHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit){};

  virtual void frameHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
                            const FrameHeader_t* frameHeader, const PackedHit_t* packedHits){};

  virtual void trailerHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
                              const CrateTrailer_t* crateTrailer, const Diagnostic_t* diagnostics){};

  bool processHBF();
  bool processDRM();

  /** decoder private functions and data members **/
  inline void decoderRewind() { mDecoderPointer = reinterpret_cast<uint32_t*>(mDecoderBuffer); };

  char* mDecoderBuffer = nullptr;
  long mDecoderBufferSize;
  uint32_t* mDecoderPointer = nullptr;
  uint32_t* mDecoderPointerMax = nullptr;
  uint32_t* mDecoderPointerNext = nullptr;
  o2::header::RAWDataHeader* mDecoderRDH;
  bool mDecoderVerbose = false;
  bool mDecoderError = false;
  bool mDecoderFatal = false;
  char mDecoderSaveBuffer[1048576];
  uint32_t mDecoderSaveBufferDataSize = 0;
  uint32_t mDecoderSaveBufferDataLeft = 0;
};

} // namespace compressed
} // namespace tof
} // namespace o2

#endif /** O2_TOF_DECODERBASE **/
