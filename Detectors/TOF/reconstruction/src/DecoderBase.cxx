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

/// @file   Decoder.cxx
/// @author Roberto Preghenella
/// @since  2020-02-24
/// @brief  TOF compressed data decoder base class

#include "TOFReconstruction/DecoderBase.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RDHUtils.h"

#include <cstring>
#include <iostream>

// o2::ctf::CTFIOSize iosize;
#define ENCODER_PARANOID
// o2::ctf::CTFIOSize iosize;
#define ENCODER_VERBOSE

#ifdef DECODER_PARANOID
#warning "Building code with DecoderParanoid option. This may limit the speed."
#endif
#ifdef DECODER_VERBOSE
#warning "Building code with DecoderVerbose option. This may limit the speed."
#endif

#define colorReset "\033[0m"
#define colorRed "\033[1;31m"
#define colorGreen "\033[1;32m"
#define colorYellow "\033[1;33m"
#define colorBlue "\033[1;34m"

namespace o2
{
namespace tof
{
namespace compressed
{

using RDHUtils = o2::raw::RDHUtils;

template <typename RDH>
bool DecoderBaseT<RDH>::processHBF()
{

#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    std::cout << colorBlue
              << "--- PROCESS HBF"
              << colorReset
              << std::endl;
  }
#endif

  if (mDecoderBufferSize <= 0) {
    std::cout << colorRed
              << " got an empty buffer, do nothing"
              << colorReset
              << std::endl;
    return true;
  }

  mDecoderRDH = reinterpret_cast<const RDH*>(mDecoderPointer);
  auto rdh = mDecoderRDH;

  int iterations = 0;
  /** loop until RDH close **/
  while (!RDHUtils::getStop(*rdh)) {
    iterations++;
    if (iterations > 5) {
      return true;
    }
#ifdef DECODER_VERBOSE
    if (mDecoderVerbose) {
      std::cout << colorBlue
                << "--- RDH open/continue detected"
                << colorReset
                << std::endl;
      o2::raw::RDHUtils::printRDH(*rdh);
    }
#endif

    /** rdh handler **/
    rdhHandler(rdh);

    auto headerSize = RDHUtils::getHeaderSize(*rdh);
    auto memorySize = RDHUtils::getMemorySize(*rdh);
    auto offsetToNext = RDHUtils::getOffsetToNext(*rdh);
    auto drmPayload = memorySize - headerSize;

    bool isValidRDH = RDHUtils::checkRDH(*rdh, false);

    /** copy DRM payload to save buffer **/
    if (isValidRDH && drmPayload > 0) {
      std::memcpy(mDecoderSaveBuffer + mDecoderSaveBufferDataSize, reinterpret_cast<const char*>(rdh) + headerSize, drmPayload);
      mDecoderSaveBufferDataSize += drmPayload;
    }
#ifdef DECODER_VERBOSE
    else {
      if (mDecoderVerbose) {
        RDHUtils::checkRDH(*rdh); // verbose
      }
    }
#endif

    /** move to next RDH **/
    rdh = reinterpret_cast<const RDH*>(reinterpret_cast<const char*>(rdh) + offsetToNext);

    /** check next RDH is within buffer **/
    if (reinterpret_cast<const char*>(rdh) < mDecoderBuffer + mDecoderBufferSize) {
      continue;
    }

    /** otherwise return **/
    return true;
  }

#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    std::cout << colorBlue
              << "--- RDH close detected"
              << colorReset
              << std::endl;
    o2::raw::RDHUtils::printRDH(*rdh);
  }
#endif

  /** process DRM data **/
  mDecoderPointer = reinterpret_cast<const uint32_t*>(mDecoderSaveBuffer);
  mDecoderPointerMax = reinterpret_cast<const uint32_t*>(mDecoderSaveBuffer + mDecoderSaveBufferDataSize);
  while (mDecoderPointer < mDecoderPointerMax) {
    if (processDRM()) {
      break;
    }
  }
  mDecoderSaveBufferDataSize = 0;

  /** rdh handler **/
  rdhHandler(rdh);

#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    std::cout << colorBlue
              << "--- END PROCESS HBF"
              << colorReset
              << std::endl;
  }
#endif

  /** move to next RDH **/
  mDecoderPointer = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(rdh) + RDHUtils::getOffsetToNext(*rdh));

  /** check next RDH is within buffer **/
  if (reinterpret_cast<const char*>(mDecoderPointer) < mDecoderBuffer + mDecoderBufferSize) {
    return false;
  }

  /** otherwise return **/
  return true;
}

template <typename RDH>
bool DecoderBaseT<RDH>::processDRM()
{

#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    std::cout << colorBlue << "--- PROCESS DRM"
              << colorReset
              << std::endl;
  }
#endif

  if ((*mDecoderPointer & 0x80000000) != 0x80000000) {
#ifdef DECODER_VERBOSE
    if (mDecoderVerbose) {
      printf(" %08x [ERROR] \n ", *mDecoderPointer);
    }
#endif
    return true;
  }

  /** crate header detected **/
  auto crateHeader = reinterpret_cast<const CrateHeader_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    printf(" %08x CrateHeader          (drmID=%d) \n ", *mDecoderPointer, crateHeader->drmID);
  }
#endif
  mDecoderPointer++;

  /** crate orbit expected **/
  auto crateOrbit = reinterpret_cast<const CrateOrbit_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    printf(" %08x CrateOrbit           (orbit=0x%08x) \n ", *mDecoderPointer, crateOrbit->orbitID);
  }
#endif
  mDecoderPointer++;

  /** header handler **/
  headerHandler(crateHeader, crateOrbit);

  while (true) {

    /** crate trailer detected **/
    if (*mDecoderPointer & 0x80000000) {
      auto crateTrailer = reinterpret_cast<const CrateTrailer_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
      if (mDecoderVerbose) {
        printf(" %08x CrateTrailer         (numberOfDiagnostics=%d, numberOfErrors=%d) \n ", *mDecoderPointer, crateTrailer->numberOfDiagnostics, crateTrailer->numberOfErrors);
      }
#endif
      mDecoderPointer++;
      auto diagnostics = reinterpret_cast<const Diagnostic_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
      if (mDecoderVerbose) {
        for (int i = 0; i < crateTrailer->numberOfDiagnostics; ++i) {
          auto diagnostic = reinterpret_cast<const Diagnostic_t*>(mDecoderPointer + i);
          printf(" %08x Diagnostic           (slotId=%d) \n ", *(mDecoderPointer + i), diagnostic->slotID);
        }
      }
#endif
      mDecoderPointer += crateTrailer->numberOfDiagnostics;
      auto errors = reinterpret_cast<const Error_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
      if (mDecoderVerbose) {
        for (int i = 0; i < crateTrailer->numberOfErrors; ++i) {
          auto error = reinterpret_cast<const Error_t*>(mDecoderPointer + i);
          printf(" %08x Error                (slotId=%d) \n ", *(mDecoderPointer + i), error->slotID);
        }
      }
#endif
      mDecoderPointer += crateTrailer->numberOfErrors;

      /** trailer handler **/
      trailerHandler(crateHeader, crateOrbit, crateTrailer, diagnostics, errors);

      return false;
    }

    /** frame header detected **/
    auto frameHeader = reinterpret_cast<const FrameHeader_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
    if (mDecoderVerbose) {
      printf(" %08x FrameHeader          (numberOfHits=%d) \n ", *mDecoderPointer, frameHeader->numberOfHits);
    }
#endif
    mDecoderPointer++;
    auto packedHits = reinterpret_cast<const PackedHit_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
    if (mDecoderVerbose) {
      for (int i = 0; i < frameHeader->numberOfHits; ++i) {
        auto packedHit = reinterpret_cast<const PackedHit_t*>(mDecoderPointer + 1);
        printf(" %08x PackedHit            (tdcID=%d) \n ", *(mDecoderPointer + 1), packedHit->tdcID);
        packedHits++;
      }
    }
#endif
    mDecoderPointer += frameHeader->numberOfHits;

    /** frame handler **/
    frameHandler(crateHeader, crateOrbit, frameHeader, packedHits);
  }

  /** should never reach here **/
  return false;
}

template class DecoderBaseT<o2::header::RAWDataHeaderV4>;
template class DecoderBaseT<o2::header::RAWDataHeaderV6>;

} // namespace compressed
} // namespace tof
} // namespace o2
