// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include <cstring>
#include <iostream>

//#define DECODER_PARANOID
//#define DECODER_VERBOSE

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

bool DecoderBase::processHBF()
{

#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    std::cout << colorBlue
              << "--- PROCESS HBF"
              << colorReset
              << std::endl;
  }
#endif

  mDecoderRDH = reinterpret_cast<o2::header::RAWDataHeader*>(mDecoderPointer);
  auto rdh = mDecoderRDH;

  /** loop until RDH close **/
  while (!rdh->stop) {

#ifdef DECODER_VERBOSE
    if (mDecoderVerbose) {
      std::cout << colorBlue
                << "--- RDH open/continue detected"
                << colorReset
                << std::endl;
      o2::raw::HBFUtils::printRDH(*rdh);
    }
#endif

    /** rdh handler **/
    rdhHandler(rdh);

    auto headerSize = rdh->headerSize;
    auto memorySize = rdh->memorySize;
    auto offsetToNext = rdh->offsetToNext;
    auto drmPayload = memorySize - headerSize;

    /** copy DRM payload to save buffer **/
    std::memcpy(mDecoderSaveBuffer + mDecoderSaveBufferDataSize, reinterpret_cast<char*>(rdh) + headerSize, drmPayload);
    mDecoderSaveBufferDataSize += drmPayload;

    /** move to next RDH **/
    rdh = reinterpret_cast<o2::header::RAWDataHeader*>(reinterpret_cast<char*>(rdh) + offsetToNext);

    /** check next RDH is within buffer **/
    if (reinterpret_cast<char*>(rdh) < mDecoderBuffer + mDecoderBufferSize)
      continue;

    /** otherwise return **/
    return true;
  }

#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    std::cout << colorBlue
              << "--- RDH close detected"
              << colorReset
              << std::endl;
    o2::raw::HBFUtils::printRDH(*rdh);
  }
#endif

  /** process DRM data **/
  mDecoderPointer = reinterpret_cast<uint32_t*>(mDecoderSaveBuffer);
  mDecoderPointerMax = reinterpret_cast<uint32_t*>(mDecoderSaveBuffer + mDecoderSaveBufferDataSize);
  while (mDecoderPointer < mDecoderPointerMax) {
    if (processDRM())
      break;
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
  mDecoderPointer = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(rdh) + rdh->offsetToNext);

  /** check next RDH is within buffer **/
  if (reinterpret_cast<char*>(mDecoderPointer) < mDecoderBuffer + mDecoderBufferSize)
    return false;

  /** otherwise return **/
  return true;
}

bool DecoderBase::processDRM()
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
  auto crateHeader = reinterpret_cast<CrateHeader_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    printf(" %08x CrateHeader          (drmID=%d) \n ", *mDecoderPointer, crateHeader->drmID);
  }
#endif
  mDecoderPointer++;

  /** crate orbit expected **/
  auto crateOrbit = reinterpret_cast<CrateOrbit_t*>(mDecoderPointer);
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
      auto crateTrailer = reinterpret_cast<CrateTrailer_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
      if (mDecoderVerbose) {
        printf(" %08x CrateTrailer         (numberOfDiagnostics=%d) \n ", *mDecoderPointer, crateTrailer->numberOfDiagnostics);
      }
#endif
      mDecoderPointer++;
      auto diagnostics = reinterpret_cast<Diagnostic_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
      if (mDecoderVerbose) {
        for (int i = 0; i < crateTrailer->numberOfDiagnostics; ++i) {
          auto diagnostic = reinterpret_cast<Diagnostic_t*>(mDecoderPointer + i);
          printf(" %08x Diagnostic           (slotId=%d) \n ", *(mDecoderPointer + i), diagnostic->slotID);
        }
      }
#endif
      mDecoderPointer += crateTrailer->numberOfDiagnostics;

      /** trailer handler **/
      trailerHandler(crateHeader, crateOrbit, crateTrailer, diagnostics);

      return false;
    }

    /** frame header detected **/
    auto frameHeader = reinterpret_cast<FrameHeader_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
    if (mDecoderVerbose) {
      printf(" %08x FrameHeader          (numberOfHits=%d) \n ", *mDecoderPointer, frameHeader->numberOfHits);
    }
#endif
    mDecoderPointer++;
    auto packedHits = reinterpret_cast<PackedHit_t*>(mDecoderPointer);
#ifdef DECODER_VERBOSE
    if (mDecoderVerbose) {
      for (int i = 0; i < frameHeader->numberOfHits; ++i) {
        auto packedHit = reinterpret_cast<PackedHit_t*>(mDecoderPointer + 1);
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

} // namespace compressed
} // namespace tof
} // namespace o2
