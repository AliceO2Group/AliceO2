// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Compressor.h
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  TOF raw data compressor

#include "TOFCompression/Compressor.h"
#include "TOFBase/Geo.h"
#include "DetectorsRaw/RDHUtils.h"

#include <cstring>
#include <iostream>

//#define DECODER_PARANOID
//#define CHECKER_COUNTER

#ifdef DECODER_PARANOID
#warning "Building code with DecoderParanoid option. This may limit the speed."
#endif
#ifdef CHECKER_COUNTER
#warning "Building code with CheckerCounter option. This may limit the speed."
#endif

#define colorReset "\033[0m"
#define colorRed "\033[1;31m"
#define colorGreen "\033[1;32m"
#define colorYellow "\033[1;33m"
#define colorBlue "\033[1;34m"

// decoding macros
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
#define IS_TRM_CHAIN_TRAILER(x, c) ((x & 0xF0000000) == (c == 0 ? 0x10000000 : 0x30000000))
#define IS_TDC_ERROR(x) ((x & 0xF0000000) == 0x60000000)
#define IS_FILLER(x) ((x & 0xFFFFFFFF) == 0x70000000)
#define IS_TDC_HIT(x) ((x & 0x80000000) == 0x80000000)
#define IS_TDC_HIT_LEADING(x) ((x & 0xA0000000) == 0xA0000000)
#define IS_TDC_HIT_TRAILING(x) ((x & 0xC0000000) == 0xC0000000)
#define IS_DRM_TEST_WORD(x) ((x & 0xF000000F) == 0xE000000F)

// DRM getters
#define GET_DRMDATAHEADER_DRMID(x) DRM_DRMID(x)
#define GET_DRMDATAHEADER_EVENTWORDS(x) DRM_EVWORDS(x)
#define GET_DRMHEADW1_PARTSLOTMASK(x) DRM_SLOTID(x)
#define GET_DRMHEADW1_CLOCKSTATUS(x) DRM_CLKFLG(x)
#define GET_DRMHEADW1_DRMHVERSION(x) DRM_VERSID(x)
#define GET_DRMHEADW1_DRMHSIZE(x) DRM_HSIZE(x)
#define GET_DRMHEADW2_ENASLOTMASK(x) DRM_ENABLEID(x)
#define GET_DRMHEADW2_FAULTSLOTMASK(x) DRM_FAULTID(x)
#define GET_DRMHEADW2_READOUTTIMEOUT(x) DRM_RTMO(x)
#define GET_DRMHEADW3_GBTBUNCHCNT(x) DRM_BCGBT(x)
#define GET_DRMHEADW3_LOCBUNCHCNT(x) DRM_BCLOC(x)
#define GET_DRMHEADW5_EVENTCRC(x) DRM_EVCRC(x)
#define GET_DRMDATATRAILER_LOCEVCNT(x) DRM_LOCEVCNT(x)

// TRM getter
#define GET_TRMDATAHEADER_SLOTID(x) TOF_GETGEO(x)
#define GET_TRMDATAHEADER_EVENTCNT(x) TRM_EVCNT_GH(x)
#define GET_TRMDATAHEADER_EVENTWORDS(x) TRM_EVWORDS(x)
#define GET_TRMDATAHEADER_EMPTYBIT(x) TRM_EMPTYBIT(x)
#define GET_TRMDATATRAILER_LUTERRORBIT(x) TRM_LUTERRBIT(x)
#define GET_TRMDATATRAILER_EVENTCRC(x) TRM_EVCRC2(x)

// TRM Chain getters
#define GET_TRMCHAINHEADER_SLOTID(x) TOF_GETGEO(x)
#define GET_TRMCHAINHEADER_BUNCHCNT(x) TRM_BUNCHID(x)
#define GET_TRMCHAINTRAILER_EVENTCNT(x) TRM_EVCNT_CT(x)
#define GET_TRMCHAINTRAILER_STATUS(x) TRM_CHAINSTAT(x)

// TDC getters
#define GET_TRMDATAHIT_TIME(x) TRM_TIME(x)
#define GET_TRMDATAHIT_CHANID(x) TRM_CHANID(x)
#define GET_TRMDATAHIT_TDCID(x) TRM_TDCID(x)
#define GET_TRMDATAHIT_EBIT(x) ((x & 0x10000000) >> 28)

namespace o2
{
namespace tof
{

template <typename RDH, bool verbose>
bool Compressor<RDH, verbose>::processHBF()
{

  if (verbose && mDecoderVerbose) {
    std::cout << colorBlue
              << "--- PROCESS HBF"
              << colorReset
              << std::endl;
  }

  mDecoderRDH = reinterpret_cast<const RDH*>(mDecoderPointer);
  mEncoderRDH = reinterpret_cast<RDH*>(mEncoderPointer);
  auto rdh = mDecoderRDH;

  /** loop until RDH close **/
  while (!rdh->stop) {

    if (verbose && mDecoderVerbose) {
      std::cout << colorBlue
                << "--- RDH open/continue detected"
                << colorReset
                << std::endl;
      o2::raw::RDHUtils::printRDH(*rdh);
    }

    auto headerSize = rdh->headerSize;
    auto memorySize = rdh->memorySize;
    auto offsetToNext = rdh->offsetToNext;
    auto drmPayload = memorySize - headerSize;

    /** copy DRM payload to save buffer **/
    std::memcpy(mDecoderSaveBuffer + mDecoderSaveBufferDataSize, reinterpret_cast<const char*>(rdh) + headerSize, drmPayload);
    mDecoderSaveBufferDataSize += drmPayload;

    /** move to next RDH **/
    rdh = reinterpret_cast<const RDH*>(reinterpret_cast<const char*>(rdh) + offsetToNext);

    /** check next RDH is within buffer **/
    if (reinterpret_cast<const char*>(rdh) < mDecoderBuffer + mDecoderBufferSize) {
      continue;
    }

    /** otherwise return **/
    return true;
  }

  if (verbose && mDecoderVerbose) {
    std::cout << colorBlue
              << "--- RDH close detected"
              << colorReset
              << std::endl;
    o2::raw::RDHUtils::printRDH(*rdh);
  }

  /** copy RDH open to encoder buffer **/
  std::memcpy(mEncoderPointer, mDecoderRDH, mDecoderRDH->headerSize);
  mEncoderPointer = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(mEncoderPointer) + rdh->headerSize);

  /** process DRM data **/
  mDecoderPointer = reinterpret_cast<const uint32_t*>(mDecoderSaveBuffer);
  mDecoderPointerMax = reinterpret_cast<const uint32_t*>(mDecoderSaveBuffer + mDecoderSaveBufferDataSize);
  while (mDecoderPointer < mDecoderPointerMax) {
    mEventCounter++;
    if (processDRM()) {            // if this breaks, we did not run the checker and the summary is not reset!
      mDecoderSummary = {nullptr}; // reset it like this, perhaps a better way can be found
      break;
    }
  }
  mDecoderSaveBufferDataSize = 0;

  /** bring encoder pointer back if fatal error **/
  if (mDecoderFatal) {
    mFatalCounter++;
    mEncoderPointer = mEncoderPointerStart;
  }

  if (mDecoderError) {
    mErrorCounter++;
  }

  /** updated encoder RDH open **/
  mEncoderRDH->memorySize = reinterpret_cast<char*>(mEncoderPointer) - reinterpret_cast<char*>(mEncoderRDH);
  mEncoderRDH->offsetToNext = mEncoderRDH->memorySize;

  /** copy RDH close to encoder buffer **/
  /** CAREFUL WITH THE PAGE COUNTER **/
  mEncoderRDH = reinterpret_cast<RDH*>(mEncoderPointer);
  std::memcpy(mEncoderRDH, rdh, rdh->headerSize);
  mEncoderRDH->memorySize = rdh->headerSize;
  mEncoderRDH->offsetToNext = mEncoderRDH->memorySize;
  mEncoderPointer = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(mEncoderPointer) + rdh->headerSize);

  if (verbose && mDecoderVerbose) {
    std::cout << colorBlue
              << "--- END PROCESS HBF"
              << colorReset
              << std::endl;
  }

  /** move to next RDH **/
  mDecoderPointer = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(rdh) + rdh->offsetToNext);

  /** check next RDH is within buffer **/
  if (reinterpret_cast<const char*>(mDecoderPointer) < mDecoderBuffer + mDecoderBufferSize) {
    return false;
  }

  /** otherwise return **/
  return true;
}

template <typename RDH, bool verbose>
bool Compressor<RDH, verbose>::processDRM()
{

  if (verbose && mDecoderVerbose) {
    std::cout << colorBlue << "--- PROCESS DRM"
              << colorReset
              << std::endl;
  }

  /** init decoder **/
  mDecoderNextWord = 1;
  mDecoderError = false;
  mDecoderFatal = false;
  mEncoderPointerStart = mEncoderPointer;

  /** check TOF Data Header **/
  if (!IS_DRM_COMMON_HEADER(*mDecoderPointer)) {
    if (verbose) {
      printf("%s %08x [ERROR] fatal error %s \n", colorRed, *mDecoderPointer, colorReset);
    }
    mDecoderFatal = true;
    return true;
  }
  mDecoderSummary.tofDataHeader = mDecoderPointer;
  if (verbose && mDecoderVerbose) {
    auto tofDataHeader = reinterpret_cast<const raw::TOFDataHeader_t*>(mDecoderPointer);
    auto bytePayload = tofDataHeader->bytePayload;
    printf(" %08x TOF Data Header       (bytePayload=%d) \n", *mDecoderPointer, bytePayload);
  }
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** TOF Orbit **/
  mDecoderSummary.tofOrbit = mDecoderPointer;
  if (verbose && mDecoderVerbose) {
    auto tofOrbit = reinterpret_cast<const raw::TOFOrbit_t*>(mDecoderPointer);
    auto orbit = tofOrbit->orbit;
    printf(" %08x TOF Orbit             (orbit=%u) \n", *mDecoderPointer, orbit);
  }
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** check DRM Data Header **/
  if (!IS_DRM_GLOBAL_HEADER(*mDecoderPointer)) {
    if (verbose) {
      printf("%s %08x [ERROR] fatal error %s \n", colorRed, *mDecoderPointer, colorReset);
    }
    mDecoderFatal = true;
    return true;
  }
  mDecoderSummary.drmDataHeader = mDecoderPointer;
  if (verbose && mDecoderVerbose) {
    auto drmDataHeader = reinterpret_cast<const raw::DRMDataHeader_t*>(mDecoderPointer);
    auto drmId = drmDataHeader->drmId;
    auto eventWords = drmDataHeader->eventWords;
    printf(" %08x DRM Data Header       (drmId=%d, eventWords=%d) \n", *mDecoderPointer, drmId, eventWords);
  }
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** DRM Header Word 1 **/
  mDecoderSummary.drmHeadW1 = mDecoderPointer;
  if (verbose && mDecoderVerbose) {
    auto drmHeadW1 = reinterpret_cast<const raw::DRMHeadW1_t*>(mDecoderPointer);
    auto partSlotMask = drmHeadW1->partSlotMask;
    auto clockStatus = drmHeadW1->clockStatus;
    auto drmHSize = drmHeadW1->drmHSize;
    printf(" %08x DRM Header Word 1     (partSlotMask=0x%03x, clockStatus=%d, drmHSize=%d) \n", *mDecoderPointer, partSlotMask, clockStatus, drmHSize);
  }
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** DRM Header Word 2 **/
  mDecoderSummary.drmHeadW2 = mDecoderPointer;
  if (verbose && mDecoderVerbose) {
    auto drmHeadW2 = reinterpret_cast<const raw::DRMHeadW2_t*>(mDecoderPointer);
    auto enaSlotMask = drmHeadW2->enaSlotMask;
    auto faultSlotMask = drmHeadW2->faultSlotMask;
    auto readoutTimeOut = drmHeadW2->readoutTimeOut;
    printf(" %08x DRM Header Word 2     (enaSlotMask=0x%03x, faultSlotMask=0x%03x, readoutTimeOut=%d) \n", *mDecoderPointer, enaSlotMask, faultSlotMask, readoutTimeOut);
  }
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** DRM Header Word 3 **/
  mDecoderSummary.drmHeadW3 = mDecoderPointer;
  if (verbose && mDecoderVerbose) {
    auto drmHeadW3 = reinterpret_cast<const raw::DRMHeadW3_t*>(mDecoderPointer);
    auto gbtBunchCnt = drmHeadW3->gbtBunchCnt;
    auto locBunchCnt = drmHeadW3->locBunchCnt;
    printf(" %08x DRM Header Word 3     (gbtBunchCnt=%d, locBunchCnt=%d) \n", *mDecoderPointer, gbtBunchCnt, locBunchCnt);
  }
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** DRM Header Word 4 **/
  mDecoderSummary.drmHeadW4 = mDecoderPointer;
  if (verbose && mDecoderVerbose) {
    printf(" %08x DRM Header Word 4   \n", *mDecoderPointer);
  }
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** DRM Header Word 5 **/
  mDecoderSummary.drmHeadW5 = mDecoderPointer;
  if (verbose && mDecoderVerbose) {
    printf(" %08x DRM Header Word 5   \n", *mDecoderPointer);
  }
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** encode Crate Header **/
  *mEncoderPointer = 0x80000000;
  *mEncoderPointer |= GET_DRMHEADW1_PARTSLOTMASK(*mDecoderSummary.drmHeadW1) << 12;
  *mEncoderPointer |= GET_DRMDATAHEADER_DRMID(*mDecoderSummary.drmDataHeader) << 24;
  *mEncoderPointer |= GET_DRMHEADW3_GBTBUNCHCNT(*mDecoderSummary.drmHeadW3);
  if (verbose && mEncoderVerbose) {
    auto crateHeader = reinterpret_cast<compressed::CrateHeader_t*>(mEncoderPointer);
    auto bunchID = crateHeader->bunchID;
    auto drmID = crateHeader->drmID;
    auto slotPartMask = crateHeader->slotPartMask;
    printf("%s %08x Crate header          (drmID=%d, bunchID=%d, slotPartMask=0x%x) %s \n", colorGreen, *mEncoderPointer, drmID, bunchID, slotPartMask, colorReset);
  }
  encoderNext();

  /** encode Crate Orbit **/
  *mEncoderPointer = *mDecoderSummary.tofOrbit;
  if (verbose && mEncoderVerbose) {
    auto crateOrbit = reinterpret_cast<compressed::CrateOrbit_t*>(mEncoderPointer);
    auto orbitID = crateOrbit->orbitID;
    printf("%s %08x Crate orbit           (orbitID=%u) %s \n", colorGreen, *mEncoderPointer, orbitID, colorReset);
  }
  encoderNext();

  /** loop over DRM payload **/
  while (true) {

    /** LTM global header detected **/
    if (IS_LTM_GLOBAL_HEADER(*mDecoderPointer)) {
      if (processLTM()) {
        return true;
      }
    }

    /** TRM Data Header detected **/
    if (IS_TRM_GLOBAL_HEADER(*mDecoderPointer) && GET_TRMDATAHEADER_SLOTID(*mDecoderPointer) > 2) {
      if (processTRM()) {
        return true;
      }
      continue;
    }

    /** DRM Data Trailer detected **/
    if (IS_DRM_GLOBAL_TRAILER(*mDecoderPointer)) {
      mDecoderSummary.drmDataTrailer = mDecoderPointer;
      if (verbose && mDecoderVerbose) {
        auto drmDataTrailer = reinterpret_cast<const raw::DRMDataTrailer_t*>(mDecoderPointer);
        auto locEvCnt = drmDataTrailer->locEvCnt;
        printf(" %08x DRM Data Trailer      (locEvCnt=%d) \n", *mDecoderPointer, locEvCnt);
      }
#ifdef DECODER_PARANOID
      if (decoderParanoid())
        return true;
#endif
      decoderNext();

      /** filler detected **/
      if (IS_FILLER(*mDecoderPointer)) {
        if (verbose && mDecoderVerbose) {
          printf(" %08x Filler \n", *mDecoderPointer);
        }
#ifdef DECODER_PARANOID
        if (decoderParanoid())
          return true;
#endif
        decoderNext();
      }

      /** encode Crate Trailer **/
      *mEncoderPointer = 0x80000000;
      *mEncoderPointer |= GET_DRMDATATRAILER_LOCEVCNT(*mDecoderSummary.drmDataTrailer) << 4;

      /** check event **/
      checkerCheck();
      *mEncoderPointer |= mCheckerSummary.nDiagnosticWords;
#if ENCODE_TDC_ERRORS
      *mEncoderPointer |= (mCheckerSummary.nTDCErrors << 16);
#endif

      if (verbose && mEncoderVerbose) {
        auto CrateTrailer = reinterpret_cast<compressed::CrateTrailer_t*>(mEncoderPointer);
        auto EventCounter = CrateTrailer->eventCounter;
        auto NumberOfDiagnostics = CrateTrailer->numberOfDiagnostics;
        auto NumberOfErrors = CrateTrailer->numberOfErrors;
        printf("%s %08x Crate trailer         (EventCounter=%d, NumberOfDiagnostics=%d, NumberOfErrors=%d) %s \n", colorGreen, *mEncoderPointer, EventCounter, NumberOfDiagnostics, NumberOfErrors, colorReset);
      }
      encoderNext();

      /** encode Diagnostic Words **/
      for (int iword = 0; iword < mCheckerSummary.nDiagnosticWords; ++iword) {
        auto itrm = (mCheckerSummary.DiagnosticWord[iword] & 0xF) - 3;
        *mEncoderPointer = mCheckerSummary.DiagnosticWord[iword];
        if (verbose && mEncoderVerbose) {
          auto Diagnostic = reinterpret_cast<compressed::Diagnostic_t*>(mEncoderPointer);
          auto slotId = Diagnostic->slotID;
          auto faultBits = Diagnostic->faultBits;
          printf("%s %08x Diagnostic            (slotId=%d, faultBits=0x%x) %s \n", colorGreen, *mEncoderPointer, slotId, faultBits, colorReset);
        }
        encoderNext();
      }

      /** encode TDC errors **/
      for (int itrm = 0; itrm < 10; ++itrm) {
        for (int ichain = 0; ichain < 2; ++ichain) {
#if ENCODE_TDC_ERRORS
          for (int ierror = 0; ierror < mDecoderSummary.trmErrors[itrm][ichain]; ++ierror) {
            *mEncoderPointer = *mDecoderSummary.trmError[itrm][ichain][ierror];
            *mEncoderPointer &= 0xFF07FFFF;
            *mEncoderPointer |= ((itrm + 3) << 19);
            *mEncoderPointer |= (ichain << 23);
            if (verbose && mEncoderVerbose) {
              auto Error = reinterpret_cast<compressed::Error_t*>(mEncoderPointer);
              auto errorFlags = Error->errorFlags;
              auto slotID = Error->slotID;
              auto chain = Error->chain;
              auto tdcID = Error->tdcID;
              printf("%s %08x Error                 (slotId=%d, chain=%d, tdcId=%d, errorFlags=0x%x) %s \n", colorGreen, *mEncoderPointer, slotID, chain, tdcID, errorFlags, colorReset);
            }
            encoderNext();
          }
#endif
          mDecoderSummary.trmErrors[itrm][ichain] = 0;
        }
      }

      mCheckerSummary.nDiagnosticWords = 0;
      mCheckerSummary.nTDCErrors = 0;

      break;
    }

    /** DRM Test Word detected **/
    if (IS_DRM_TEST_WORD(*mDecoderPointer)) {
      if (verbose && mDecoderVerbose) {
        printf(" %08x DRM Test Word \n", *mDecoderPointer);
      }
      decoderNext();
      continue;
    }

    /** decode error **/
    mDecoderError = true;
    mDecoderSummary.drmDecodeError = true;

    if (verbose && mDecoderVerbose) {
      printf("%s %08x [ERROR] trying to recover DRM decode stream %s \n", colorRed, *mDecoderPointer, colorReset);
    }

    /** decode error detected, be paranoid **/
    if (decoderParanoid()) {
      return true;
    }

    decoderNext();

  } /** end of loop over DRM payload **/

  mIntegratedBytes += getDecoderByteCounter();

  if (verbose && mDecoderVerbose) {
    std::cout << colorBlue
              << "--- END PROCESS DRM"
              << colorReset
              << std::endl;
  }

  return false;
}

template <typename RDH, bool verbose>
bool Compressor<RDH, verbose>::processLTM()
{
  /** process LTM **/

  if (verbose && mDecoderVerbose) {
    printf(" %08x LTM Global Header \n", *mDecoderPointer);
  }
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** loop over LTM payload **/
  while (true) {
    /** LTM global trailer detected **/
    if (IS_LTM_GLOBAL_TRAILER(*mDecoderPointer)) {
      if (verbose && mDecoderVerbose) {
        printf(" %08x LTM Global Trailer \n", *mDecoderPointer);
      }
#ifdef DECODER_PARANOID
      if (decoderParanoid())
        return true;
#endif
      decoderNext();
      break;
    }

    if (verbose && mDecoderVerbose) {
      printf(" %08x LTM data \n", *mDecoderPointer);
    }
#ifdef DECODER_PARANOID
    if (decoderParanoid())
      return true;
#endif
    decoderNext();
  }

  /** success **/
  return false;
}

template <typename RDH, bool verbose>
bool Compressor<RDH, verbose>::processTRM()
{
  /** process TRM **/

  uint32_t slotId = GET_TRMDATAHEADER_SLOTID(*mDecoderPointer);
  int itrm = slotId - 3;
  mDecoderSummary.trmDataHeader[itrm] = mDecoderPointer;
  if (verbose && mDecoderVerbose) {
    auto trmDataHeader = reinterpret_cast<const raw::TRMDataHeader_t*>(mDecoderPointer);
    auto eventWords = trmDataHeader->eventWords;
    auto eventCnt = trmDataHeader->eventCnt;
    auto emptyBit = trmDataHeader->emptyBit;
    printf(" %08x TRM Data Header       (slotId=%u, eventWords=%d, eventCnt=%d, emptyBit=%01x) \n", *mDecoderPointer, slotId, eventWords, eventCnt, emptyBit);
  }
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** loop over TRM payload **/
  while (true) {

    /** TRM Chain-A Header detected **/
    if (IS_TRM_CHAINA_HEADER(*mDecoderPointer) && GET_TRMCHAINHEADER_SLOTID(*mDecoderPointer) == slotId) {
      if (processTRMchain(itrm, 0)) {
        return true;
      }
    }

    /** TRM Chain-B Header detected **/
    if (IS_TRM_CHAINB_HEADER(*mDecoderPointer) && GET_TRMCHAINHEADER_SLOTID(*mDecoderPointer) == slotId) {
      if (processTRMchain(itrm, 1)) {
        return true;
      }
    }

    /** TRM Data Trailer detected **/
    if (IS_TRM_GLOBAL_TRAILER(*mDecoderPointer)) {
      mDecoderSummary.trmDataTrailer[itrm] = mDecoderPointer;
      if (verbose && mDecoderVerbose) {
        auto trmDataTrailer = reinterpret_cast<const raw::TRMDataTrailer_t*>(mDecoderPointer);
        auto eventCRC = trmDataTrailer->eventCRC;
        auto lutErrorBit = trmDataTrailer->lutErrorBit;
        printf(" %08x TRM Data Trailer      (slotId=%u, eventCRC=%d, lutErrorBit=%d) \n", *mDecoderPointer, slotId, eventCRC, lutErrorBit);
      }
#ifdef DECODER_PARANOID
      if (decoderParanoid())
        return true;
#endif
      decoderNext();

      /** filler detected **/
      if (IS_FILLER(*mDecoderPointer)) {
        if (verbose && mDecoderVerbose) {
          printf(" %08x Filler \n", *mDecoderPointer);
        }
#ifdef DECODER_PARANOID
        if (decoderParanoid())
          return true;
#endif
        decoderNext();
      }

      /** encoder Spider **/
      if (mDecoderSummary.hasHits[itrm][0] || mDecoderSummary.hasHits[itrm][1]) {
        encoderSpider(itrm);
      }

      /** success **/
      return false;
    }

    /** decode error **/
    mDecoderError = true;
    mDecoderSummary.trmDecodeError[itrm] = true;
    if (verbose && mDecoderVerbose) {
      printf("%s %08x [ERROR] breaking TRM decode stream %s \n", colorRed, *mDecoderPointer, colorReset);
    }
    /** decode error detected, be paranoid **/
    if (decoderParanoid()) {
      return true;
    }

    decoderNext();
    return false;

  } /** end of loop over TRM payload **/

  /** never reached **/
  return false;
}

template <typename RDH, bool verbose>
bool Compressor<RDH, verbose>::processTRMchain(int itrm, int ichain)
{
  /** process TRM chain **/

  int slotId = itrm + 3;

  mDecoderSummary.trmChainHeader[itrm][ichain] = mDecoderPointer;
  mDecoderSummary.hasHits[itrm][ichain] = false;
  mDecoderSummary.hasErrors[itrm][ichain] = false;
  if (verbose && mDecoderVerbose) {
    auto trmChainHeader = reinterpret_cast<const raw::TRMChainHeader_t*>(mDecoderPointer);
    auto bunchCnt = trmChainHeader->bunchCnt;
    printf(" %08x TRM Chain-%c Header    (slotId=%u, bunchCnt=%d) \n", *mDecoderPointer, ichain == 0 ? 'A' : 'B', slotId, bunchCnt);
  }
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** loop over TRM Chain payload **/
  while (true) {
    /** TDC hit detected **/
    if (IS_TDC_HIT(*mDecoderPointer)) {
      mDecoderSummary.hasHits[itrm][ichain] = true;
      auto itdc = GET_TRMDATAHIT_TDCID(*mDecoderPointer);
      auto ihit = mDecoderSummary.trmDataHits[ichain][itdc];
      mDecoderSummary.trmDataHit[ichain][itdc][ihit] = mDecoderPointer;
      mDecoderSummary.trmDataHits[ichain][itdc]++;
      if (verbose && mDecoderVerbose) {
        auto trmDataHit = reinterpret_cast<const raw::TRMDataHit_t*>(mDecoderPointer);
        auto time = trmDataHit->time;
        auto chanId = trmDataHit->chanId;
        auto tdcId = trmDataHit->tdcId;
        auto dataId = trmDataHit->dataId;
        printf(" %08x TRM Data Hit          (time=%d, chanId=%d, tdcId=%d, dataId=0x%x) \n", *mDecoderPointer, time, chanId, tdcId, dataId);
      }
#ifdef DECODER_PARANOID
      if (decoderParanoid())
        return true;
#endif
      decoderNext();
      continue;
    }

    /** TDC error detected **/
    if (IS_TDC_ERROR(*mDecoderPointer)) {
      mDecoderSummary.hasErrors[itrm][ichain] = true;
      auto ierror = mDecoderSummary.trmErrors[itrm][ichain];
      mDecoderSummary.trmError[itrm][ichain][ierror] = mDecoderPointer;
      mDecoderSummary.trmErrors[itrm][ichain]++;
      if (verbose && mDecoderVerbose) {
        printf("%s %08x TDC error %s \n", colorRed, *mDecoderPointer, colorReset);
      }
#ifdef DECODER_PARANOID
      if (decoderParanoid())
        return true;
#endif
      decoderNext();
      continue;
    }

    /** TRM Chain Trailer detected **/
    if (IS_TRM_CHAIN_TRAILER(*mDecoderPointer, ichain)) {
      mDecoderSummary.trmChainTrailer[itrm][ichain] = mDecoderPointer;
      if (verbose && mDecoderVerbose) {
        auto trmChainTrailer = reinterpret_cast<const raw::TRMChainTrailer_t*>(mDecoderPointer);
        auto eventCnt = trmChainTrailer->eventCnt;
        printf(" %08x TRM Chain-A Trailer   (slotId=%u, eventCnt=%d) \n", *mDecoderPointer, slotId, eventCnt);
      }
#ifdef DECODER_PARANOID
      if (decoderParanoid())
        return true;
#endif
      decoderNext();
      break;
    }

    /** decode error **/
    mDecoderError = true;
    mDecoderSummary.trmDecodeError[itrm] = true;
    if (verbose && mDecoderVerbose) {
      printf("%s %08x [ERROR] breaking TRM Chain-%c decode stream %s \n", colorRed, *mDecoderPointer, ichain == 0 ? 'A' : 'B', colorReset);
    }
    /** decode error detected, be paranoid **/
    if (decoderParanoid()) {
      return true;
    }

    decoderNext();
    break;

  } /** end of loop over TRM chain payload **/

  /** success **/
  return false;
}

template <typename RDH, bool verbose>
bool Compressor<RDH, verbose>::decoderParanoid()
{
  /** decoder paranoid **/

  if (mDecoderPointer >= mDecoderPointerMax) {
    printf("%s %08x [ERROR] fatal error: beyond memory size %s \n", colorRed, *mDecoderPointer, colorReset);
    mDecoderFatal = true;
    return true;
  }
  return false;
}

template <typename RDH, bool verbose>
void Compressor<RDH, verbose>::encoderSpider(int itrm)
{
  /** encoder spider **/

  int slotId = itrm + 3;

  /** reset packed hits counter **/
  int firstFilledFrame = 255;
  int lastFilledFrame = 0;

  /** loop over TRM chains **/
  for (int ichain = 0; ichain < 2; ++ichain) {

    if (!mDecoderSummary.hasHits[itrm][ichain]) {
      continue;
    }

    /** loop over TDCs **/
    for (int itdc = 0; itdc < 15; ++itdc) {

      auto nhits = mDecoderSummary.trmDataHits[ichain][itdc];
      if (nhits == 0) {
        continue;
      }

      /** loop over hits **/
      for (int ihit = 0; ihit < nhits; ++ihit) {

        auto lhit = *mDecoderSummary.trmDataHit[ichain][itdc][ihit];
        if (!IS_TDC_HIT_LEADING(lhit)) { // must be a leading hit
          continue;
        }

        auto chan = GET_TRMDATAHIT_CHANID(lhit);
        auto hitTime = GET_TRMDATAHIT_TIME(lhit);
        auto eBit = GET_TRMDATAHIT_EBIT(lhit);
        uint32_t totWidth = 0;

        // check next hits for packing
        for (int jhit = ihit + 1; jhit < nhits; ++jhit) {
          auto thit = *mDecoderSummary.trmDataHit[ichain][itdc][jhit];
          if (IS_TDC_HIT_TRAILING(thit) && GET_TRMDATAHIT_CHANID(thit) == chan) {      // must be a trailing hit from same channel
            totWidth = (GET_TRMDATAHIT_TIME(thit) - hitTime) / Geo::RATIO_TOT_TDC_BIN; // compute TOT
            lhit = 0x0;                                                                // mark as used
            break;
          }
        }

        auto iframe = hitTime >> 13;
        auto phit = mSpiderSummary.nFramePackedHits[iframe];

        mSpiderSummary.FramePackedHit[iframe][phit] = 0x00000000;
        mSpiderSummary.FramePackedHit[iframe][phit] |= (totWidth & 0x7FF) << 0;
        mSpiderSummary.FramePackedHit[iframe][phit] |= (hitTime & 0x1FFF) << 11;
        mSpiderSummary.FramePackedHit[iframe][phit] |= chan << 24;
        mSpiderSummary.FramePackedHit[iframe][phit] |= itdc << 27;
        mSpiderSummary.FramePackedHit[iframe][phit] |= ichain << 31;
        mSpiderSummary.nFramePackedHits[iframe]++;

        if (iframe < firstFilledFrame) {
          firstFilledFrame = iframe;
        }
        if (iframe > lastFilledFrame) {
          lastFilledFrame = iframe;
        }
      }

      mDecoderSummary.trmDataHits[ichain][itdc] = 0;
    }
  }

  /** loop over frames **/
  for (int iframe = firstFilledFrame; iframe < lastFilledFrame + 1; iframe++) {

    /** check if frame is empty **/
    if (mSpiderSummary.nFramePackedHits[iframe] == 0) {
      continue;
    }

    // encode Frame Header
    *mEncoderPointer = 0x00000000;
    *mEncoderPointer |= slotId << 24;
    *mEncoderPointer |= iframe << 16;
    *mEncoderPointer |= mSpiderSummary.nFramePackedHits[iframe];
    if (verbose && mEncoderVerbose) {
      auto FrameHeader = reinterpret_cast<const compressed::FrameHeader_t*>(mEncoderPointer);
      auto NumberOfHits = FrameHeader->numberOfHits;
      auto FrameID = FrameHeader->frameID;
      auto TRMID = FrameHeader->trmID;
      printf("%s %08x Frame header          (TRMID=%d, FrameID=%d, NumberOfHits=%d) %s \n", colorGreen, *mEncoderPointer, TRMID, FrameID, NumberOfHits, colorReset);
    }
    encoderNext();

    // packed hits
    for (int ihit = 0; ihit < mSpiderSummary.nFramePackedHits[iframe]; ++ihit) {
      *mEncoderPointer = mSpiderSummary.FramePackedHit[iframe][ihit];
      if (verbose && mEncoderVerbose) {
        auto PackedHit = reinterpret_cast<const compressed::PackedHit_t*>(mEncoderPointer);
        auto Chain = PackedHit->chain;
        auto TDCID = PackedHit->tdcID;
        auto Channel = PackedHit->channel;
        auto Time = PackedHit->time;
        auto TOT = PackedHit->tot;
        printf("%s %08x Packed hit            (Chain=%d, TDCID=%d, Channel=%d, Time=%d, TOT=%d) %s \n", colorGreen, *mEncoderPointer, Chain, TDCID, Channel, Time, TOT, colorReset);
      }
      encoderNext();
    }

    mSpiderSummary.nFramePackedHits[iframe] = 0;
  }
}

template <typename RDH, bool verbose>
bool Compressor<RDH, verbose>::checkerCheck()
{
  /** checker check **/

  mCheckerSummary.nDiagnosticWords = 0;

  if (verbose && mCheckerVerbose) {
    std::cout << colorBlue
              << "--- CHECK EVENT"
              << colorReset
              << std::endl;
  }

  /** increment check counter **/
  //    mCheckerCounter++;

  /** check TOF Data Header **/

  /** check DRM **/
  mCheckerSummary.DiagnosticWord[0] = 0x00000001;

  /** check DRM Data Header **/
  if (verbose && mCheckerVerbose) {
    printf(" --- Checking DRM Data Header: %p \n", mDecoderSummary.drmDataHeader);
  }
  if (!mDecoderSummary.drmDataHeader) {
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_HEADER_MISSING;
    if (verbose && mCheckerVerbose) {
      printf(" Missing DRM Data Header \n");
    }
    mDecoderSummary = {nullptr};
    mCheckerSummary.nDiagnosticWords++;
    for (int itrm = 0; itrm < 10; ++itrm) {
      mDecoderSummary.trmDataHeader[itrm] = nullptr;
      mDecoderSummary.trmDataTrailer[itrm] = nullptr;
      for (int ichain = 0; ichain < 2; ++ichain) {
        mDecoderSummary.trmChainHeader[itrm][ichain] = nullptr;
        mDecoderSummary.trmChainTrailer[itrm][ichain] = nullptr;
        mDecoderSummary.trmErrors[itrm][ichain] = 0;
        mDecoderSummary.trmErrors[itrm][ichain] = 0;
      }
    }
    return true;
  }

  /** check DRM decode error **/
  if (mDecoderSummary.drmDecodeError) {
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_DECODE_ERROR;
    if (verbose && mCheckerVerbose) {
      printf(" DRM decode error \n");
    }
    mDecoderSummary.drmDecodeError = false;
  }

  /** check DRM Data Trailer **/
  if (verbose && mCheckerVerbose) {
    printf(" --- Checking DRM Data Trailer: %p \n", mDecoderSummary.drmDataTrailer);
  }
  if (!mDecoderSummary.drmDataTrailer) {
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_TRAILER_MISSING;
    if (verbose && mCheckerVerbose) {
      printf(" Missing DRM Data Trailer \n");
    }
    mDecoderSummary = {nullptr};
    mCheckerSummary.nDiagnosticWords++;

    return true;
  }

  /** get DRM relevant data **/
  uint32_t partSlotMask = GET_DRMHEADW1_PARTSLOTMASK(*mDecoderSummary.drmHeadW1) & 0x7FE; // remove LTM bit
  uint32_t enaSlotMask = GET_DRMHEADW2_ENASLOTMASK(*mDecoderSummary.drmHeadW2) & 0x7FE;   // remove LTM bit
  uint32_t gbtBunchCnt = GET_DRMHEADW3_GBTBUNCHCNT(*mDecoderSummary.drmHeadW3);
  uint32_t locEvCnt = GET_DRMDATATRAILER_LOCEVCNT(*mDecoderSummary.drmDataTrailer);

  /** check RDH **/
  if (!mDecoderCONET) {
    checkerCheckRDH();
  }

  /** check enable/participating mask **/
  if (verbose && mCheckerVerbose) {
    printf(" --- Checking Enable/participating mask: %03x/%03x \n", enaSlotMask, partSlotMask);
  }
  if (partSlotMask != enaSlotMask) {
    if (verbose && mCheckerVerbose) {
      printf(" Enable/participating mask differ: %03x/%03x \n", enaSlotMask, partSlotMask);
    }
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_ENAPARTMASK_DIFFER;
  }

  /** check DRM clock status **/
  if (verbose && mCheckerVerbose) {
    printf(" --- Checking DRM clock status: %d \n", GET_DRMHEADW1_CLOCKSTATUS(*mDecoderSummary.drmHeadW1));
  }
  if (GET_DRMHEADW1_CLOCKSTATUS(*mDecoderSummary.drmHeadW1) != 2) {
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_CLOCKSTATUS_WRONG;
    if (verbose && mCheckerVerbose) {
      printf("%s DRM wrong clock status: %d %s\n", colorRed, GET_DRMHEADW1_CLOCKSTATUS(*mDecoderSummary.drmHeadW1), colorReset);
    }
  }

  /** check DRM fault mask **/
  if (verbose && mCheckerVerbose) {
    printf(" --- Checking DRM fault slot mask: %x \n", GET_DRMHEADW2_FAULTSLOTMASK(*mDecoderSummary.drmHeadW2));
  }
  if (GET_DRMHEADW2_FAULTSLOTMASK(*mDecoderSummary.drmHeadW2)) {
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_FAULTSLOTMASK_NOTZERO;
    if (verbose && mCheckerVerbose) {
      printf(" DRM fault slot mask: %x \n", GET_DRMHEADW2_FAULTSLOTMASK(*mDecoderSummary.drmHeadW2));
    }
  }

  /** check DRM readout timeout **/
  if (verbose && mCheckerVerbose) {
    printf(" --- Checking DRM readout timeout: %d \n", GET_DRMHEADW2_READOUTTIMEOUT(*mDecoderSummary.drmHeadW2));
  }
  if (GET_DRMHEADW2_READOUTTIMEOUT(*mDecoderSummary.drmHeadW2)) {
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_READOUTTIMEOUT_NOTZERO;
    if (verbose && mCheckerVerbose) {
      printf(" DRM readout timeout \n");
    }
  }

  /** check DRM event words (careful with pointers because we have 64 bits extra! only for CRU data! **/
  auto drmEventWords = mDecoderSummary.drmDataTrailer - mDecoderSummary.drmDataHeader + 1;
  if (!mDecoderCONET) {
    drmEventWords -= (drmEventWords / 4) * 2;
  }
  drmEventWords -= 6;
  if (verbose && mCheckerVerbose) {
    printf(" --- Checking DRM declared/detected event words: %u/%ld \n", GET_DRMDATAHEADER_EVENTWORDS(*mDecoderSummary.drmDataHeader), drmEventWords);
  }
  if (GET_DRMDATAHEADER_EVENTWORDS(*mDecoderSummary.drmDataHeader) != drmEventWords) {
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_EVENTWORDS_MISMATCH;
    if (verbose && mCheckerVerbose) {
      printf(" DRM declared/detected event words mismatch: %u/%ld \n", GET_DRMDATAHEADER_EVENTWORDS(*mDecoderSummary.drmDataHeader), drmEventWords);
    }
  }

  /** check current diagnostic word **/
  auto iword = mCheckerSummary.nDiagnosticWords;
  if (mCheckerSummary.DiagnosticWord[iword] & 0xFFFFFFF0) {
    mCheckerSummary.nDiagnosticWords++;
    iword++;
  }

  /** check LTM **/
  mCheckerSummary.DiagnosticWord[iword] = 0x00000002;

  /** check participating LTM **/
  if (!(partSlotMask & 1)) {
    if (mDecoderSummary.ltmDataHeader != nullptr) {
      mCheckerSummary.DiagnosticWord[iword] |= diagnostic::LTM_HEADER_UNEXPECTED;
      if (verbose && mCheckerVerbose) {
        printf(" Non-participating LTM header found \n");
      }
    }
  } else {
    /** check LTM Data Header **/
    if (verbose && mCheckerVerbose) {
      printf(" --- Checking LTM Data Header: %p \n", mDecoderSummary.ltmDataHeader);
    }
    if (!mDecoderSummary.ltmDataHeader) {
      mCheckerSummary.DiagnosticWord[iword] |= diagnostic::LTM_HEADER_MISSING;
      if (verbose && mCheckerVerbose) {
        printf(" Missing LTM Data Header \n");
      }
    }

    /** check LTM Data Trailer **/
    if (verbose && mCheckerVerbose) {
      printf(" --- Checking LTM Data Trailer: %p \n", mDecoderSummary.ltmDataTrailer);
    }
    if (!mDecoderSummary.ltmDataTrailer) {
      mCheckerSummary.DiagnosticWord[iword] |= diagnostic::LTM_TRAILER_MISSING;
      if (verbose && mCheckerVerbose) {
        printf(" Missing LTM Data Trailer \n");
      }
    }
  }

  /** clear LTM summary data **/
  mDecoderSummary.ltmDataHeader = nullptr;
  mDecoderSummary.ltmDataTrailer = nullptr;

  /** loop over TRMs **/
  for (int itrm = 0; itrm < 10; ++itrm) {
    uint32_t slotId = itrm + 3;

    /** check current diagnostic word **/
    if (mCheckerSummary.DiagnosticWord[iword] & 0xFFFFFFF0) {
      mCheckerSummary.nDiagnosticWords++;
      iword++;
    }

    /** set current slot id **/
    mCheckerSummary.DiagnosticWord[iword] = slotId;

    /** check participating TRM **/
    if (!(partSlotMask & 1 << (itrm + 1))) {
      if (mDecoderSummary.trmDataHeader[itrm]) {
        mCheckerSummary.DiagnosticWord[iword] |= diagnostic::TRM_HEADER_UNEXPECTED;
        if (verbose && mCheckerVerbose) {
          printf(" Non-participating header found (slotId=%u) \n", slotId);
        }
      } else {
        continue;
      }
    }

    /** check TRM bit in DRM fault mask **/
    if (GET_DRMHEADW2_FAULTSLOTMASK(*mDecoderSummary.drmHeadW2) & 1 << (itrm + 1)) {
      mCheckerSummary.DiagnosticWord[iword] |= diagnostic::TRM_FAULTSLOTBIT_NOTZERO;
      if (verbose && mCheckerVerbose) {
        printf(" Fault slot bit set (slotId=%u) \n", slotId);
      }
    }

    /** check TRM Data Header **/
    if (!mDecoderSummary.trmDataHeader[itrm]) {
      mCheckerSummary.DiagnosticWord[iword] |= diagnostic::TRM_HEADER_MISSING;
      if (verbose && mCheckerVerbose) {
        printf(" Missing TRM Data Header (slotId=%u) \n", slotId);
      }
      mDecoderSummary.trmErrors[itrm][0] = 0;
      mDecoderSummary.trmErrors[itrm][1] = 0;
      continue;
    }

    /** check TRM decode error **/
    if (mDecoderSummary.trmDecodeError[itrm]) {
      mCheckerSummary.DiagnosticWord[iword] |= diagnostic::TRM_DECODE_ERROR;
      if (verbose && mCheckerVerbose) {
        printf(" Decode error in TRM (slotId=%u) \n", slotId);
      }
      mDecoderSummary.trmDecodeError[itrm] = false;
    }

    /** check TRM Data Trailer **/
    if (!mDecoderSummary.trmDataTrailer[itrm]) {
      mCheckerSummary.DiagnosticWord[iword] |= diagnostic::TRM_TRAILER_MISSING;
      if (verbose && mCheckerVerbose) {
        printf(" Missing TRM Trailer (slotId=%u) \n", slotId);
      }
      mDecoderSummary.trmDataHeader[itrm] = nullptr;
      mDecoderSummary.trmErrors[itrm][0] = 0;
      mDecoderSummary.trmErrors[itrm][1] = 0;
      continue;
    }

    /** increment TRM header counter **/
#ifdef CHECKER_COUNTER
    mTRMCounters[itrm].Headers++;
#endif

    /** check TRM empty flag **/
#ifdef CHECKER_COUNTER
    if (!mDecoderSummary.hasHits[itrm][0] && !mDecoderSummary.hasHits[itrm][1])
      mTRMCounters[itrm].Empty++;
#endif

    /** check TRM EventCounter **/
    uint32_t eventCnt = GET_TRMDATAHEADER_EVENTCNT(*mDecoderSummary.trmDataHeader[itrm]);
    if (eventCnt != locEvCnt % 1024) {
      mCheckerSummary.DiagnosticWord[iword] |= diagnostic::TRM_EVENTCNT_MISMATCH;
#ifdef CHECKER_COUNTER
      mTRMCounters[itrm].EventCounterMismatch++;
#endif
      if (verbose && mCheckerVerbose) {
        printf(" TRM EventCounter / DRM LocalEventCounter mismatch: %u / %u (slotId=%u) \n", eventCnt, locEvCnt, slotId);
      }
    }

    /** check TRM empty bit **/
    if (GET_TRMDATAHEADER_EMPTYBIT(*mDecoderSummary.trmDataHeader[itrm])) {
      mCheckerSummary.DiagnosticWord[iword] |= diagnostic::TRM_EMPTYBIT_NOTZERO;
#ifdef CHECKER_COUNTER
      mTRMCounters[itrm].EBit++;
#endif
      if (verbose && mCheckerVerbose) {
        printf(" TRM empty bit is on (slotId=%u) \n", slotId);
      }
    }

    /** check TRM event words (careful with pointers because we have 64 bits extra! only for CRU data! **/
    auto trmEventWords = mDecoderSummary.trmDataTrailer[itrm] - mDecoderSummary.trmDataHeader[itrm] + 1;
    if (!mDecoderCONET) {
      trmEventWords -= (trmEventWords / 4) * 2;
    }
    if (verbose && mCheckerVerbose) {
      printf(" --- Checking TRM (slotId=%u) declared/detected event words: %d/%ld \n", slotId, GET_TRMDATAHEADER_EVENTWORDS(*mDecoderSummary.trmDataHeader[itrm]), trmEventWords);
    }
    if (GET_TRMDATAHEADER_EVENTWORDS(*mDecoderSummary.trmDataHeader[itrm]) != trmEventWords) {
      mCheckerSummary.DiagnosticWord[iword] |= diagnostic::TRM_EVENTWORDS_MISMATCH;
      if (verbose && mCheckerVerbose) {
        printf(" TRM (slotId=%u) declared/detected event words mismatch: %d/%ld \n", slotId, GET_TRMDATAHEADER_EVENTWORDS(*mDecoderSummary.trmDataHeader[itrm]), trmEventWords);
      }
    }

    /** loop over TRM chains **/
    for (int ichain = 0; ichain < 2; ichain++) {

      /** check TRM Chain Header **/
      if (!mDecoderSummary.trmChainHeader[itrm][ichain]) {
        mCheckerSummary.DiagnosticWord[iword] |= (diagnostic::TRMCHAIN_HEADER_MISSING << (ichain * 8));
        if (verbose && mCheckerVerbose) {
          printf(" Missing TRM Chain Header (slotId=%u, chain=%d) \n", slotId, ichain);
        }
        mDecoderSummary.trmErrors[itrm][ichain] = 0;
        continue;
      }

      /** check TRM Chain Trailer **/
      if (!mDecoderSummary.trmChainTrailer[itrm][ichain]) {
        mCheckerSummary.DiagnosticWord[iword] |= (diagnostic::TRMCHAIN_TRAILER_MISSING << (ichain * 8));
        if (verbose && mCheckerVerbose) {
          printf(" Missing TRM Chain Trailer (slotId=%u, chain=%d) \n", slotId, ichain);
        }
        mDecoderSummary.trmChainHeader[itrm][ichain] = nullptr;
        mDecoderSummary.trmErrors[itrm][ichain] = 0;
        continue;
      }

      /** increment TRM Chain header counter **/
#ifdef CHECKER_COUNTER
      mTRMChainCounters[itrm][ichain].Headers++;
#endif

      /** check TDC errors **/
      if (mDecoderSummary.hasErrors[itrm][ichain]) {
        mCheckerSummary.DiagnosticWord[iword] |= (diagnostic::TRMCHAIN_TDCERROR_DETECTED << (ichain * 8));
        mCheckerSummary.nTDCErrors += mDecoderSummary.trmErrors[itrm][ichain];
#ifdef CHECKER_COUNTER
        mTRMChainCounters[itrm][ichain].TDCerror++;
#endif
        if (verbose && mCheckerVerbose) {
          printf(" TDC error detected (slotId=%u, chain=%d) \n", slotId, ichain);
        }
      }

      /** check TRM Chain event counter **/
      uint32_t eventCnt = GET_TRMCHAINTRAILER_EVENTCNT(*mDecoderSummary.trmChainTrailer[itrm][ichain]);
      if (eventCnt != locEvCnt) {
        mCheckerSummary.DiagnosticWord[iword] |= (diagnostic::TRMCHAIN_EVENTCNT_MISMATCH << (ichain * 8));
#ifdef CHECKER_COUNTER
        mTRMChainCounters[itrm][ichain].EventCounterMismatch++;
#endif
        if (verbose && mCheckerVerbose) {
          printf(" TRM Chain EventCounter / DRM LocalEventCounter mismatch: %u / %u (slotId=%u, chain=%d) \n", eventCnt, locEvCnt, slotId, ichain);
        }
      }

      /** check TRM Chain Status **/
      uint32_t status = GET_TRMCHAINTRAILER_STATUS(*mDecoderSummary.trmChainTrailer[itrm][ichain]);
      if (status != 0) {
        mCheckerSummary.DiagnosticWord[iword] |= (diagnostic::TRMCHAIN_STATUS_NOTZERO << (ichain * 8));
#ifdef CHECKER_COUNTER
        mTRMChainCounters[itrm][ichain].BadStatus++;
#endif
        if (verbose && mCheckerVerbose) {
          printf(" TRM Chain bad Status: %u (slotId=%u, chain=%d) \n", status, slotId, ichain);
        }
      }

      /** check TRM Chain BunchID **/
      uint32_t bunchCnt = GET_TRMCHAINHEADER_BUNCHCNT(*mDecoderSummary.trmChainHeader[itrm][ichain]);
      if (bunchCnt != gbtBunchCnt) {
        mCheckerSummary.DiagnosticWord[iword] |= (diagnostic::TRMCHAIN_BUNCHCNT_MISMATCH << (ichain * 8));
#ifdef CHECKER_COUNTER
        mTRMChainCounters[itrm][ichain].BunchIDMismatch++;
#endif
        if (verbose && mCheckerVerbose) {
          printf(" TRM Chain BunchID / DRM L0BCID mismatch: %u / %u (slotId=%u, chain=%d) \n", bunchCnt, gbtBunchCnt, slotId, ichain);
        }
      }

      /** clear TRM chain summary data **/
      mDecoderSummary.trmChainHeader[itrm][ichain] = nullptr;
      mDecoderSummary.trmChainTrailer[itrm][ichain] = nullptr;

    } /** end of loop over TRM chains **/

    /** clear TRM summary data **/
    mDecoderSummary.trmDataHeader[itrm] = nullptr;
    mDecoderSummary.trmDataTrailer[itrm] = nullptr;

  } /** end of loop over TRMs **/

  /** check current diagnostic word **/
  if (mCheckerSummary.DiagnosticWord[iword] & 0xFFFFFFF0) {
    mCheckerSummary.nDiagnosticWords++;
  }

  if (verbose && mCheckerVerbose) {
    std::cout << colorBlue
              << "--- END CHECK EVENT: " << mCheckerSummary.nDiagnosticWords << " diagnostic words"
              << colorReset
              << std::endl;
  }

  /** clear DRM summary data **/
  mDecoderSummary.tofDataHeader = nullptr;
  mDecoderSummary.drmDataHeader = nullptr;
  mDecoderSummary.drmDataTrailer = nullptr;

  return false;
}

template <typename RDH, bool verbose>
void Compressor<RDH, verbose>::checkerCheckRDH()
{
}

template <>
void Compressor<o2::header::RAWDataHeaderV4, true>::checkerCheckRDH()
{

  uint32_t orbit = *mDecoderSummary.tofOrbit;
  uint32_t drmId = GET_DRMDATAHEADER_DRMID(*mDecoderSummary.drmDataHeader);

  /** check orbit **/
  if (mCheckerVerbose) {
    printf(" --- Checking DRM/RDH orbit: %08x/%08x \n", orbit, mDecoderRDH->heartbeatOrbit);
  }
  if (orbit != mDecoderRDH->heartbeatOrbit) {
    if (mCheckerVerbose) {
      printf(" DRM/RDH orbit mismatch: %08x/%08x \n", orbit, mDecoderRDH->heartbeatOrbit);
    }
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_ORBIT_MISMATCH;
  }

  /** check FEE id **/
  if (mCheckerVerbose) {
    printf(" --- Checking DRM/RDH FEE id: %d/%d \n", drmId, mDecoderRDH->feeId & 0xFF);
  }
  if (drmId != (mDecoderRDH->feeId & 0xFF)) {
    if (mCheckerVerbose) {
      printf(" DRM/RDH FEE id mismatch: %d/%d \n", drmId, mDecoderRDH->feeId & 0xFF);
    }
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_FEEID_MISMATCH;
  }
}

template <>
void Compressor<o2::header::RAWDataHeaderV4, false>::checkerCheckRDH()
{

  uint32_t orbit = *mDecoderSummary.tofOrbit;
  uint32_t drmId = GET_DRMDATAHEADER_DRMID(*mDecoderSummary.drmDataHeader);

  /** check orbit **/
  if (orbit != mDecoderRDH->heartbeatOrbit) {
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_ORBIT_MISMATCH;
  }

  /** check FEE id **/
  if (drmId != (mDecoderRDH->feeId & 0xFF)) {
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_FEEID_MISMATCH;
  }
}

template <>
void Compressor<o2::header::RAWDataHeaderV6, true>::checkerCheckRDH()
{
  uint32_t orbit = *mDecoderSummary.tofOrbit;
  uint32_t drmId = GET_DRMDATAHEADER_DRMID(*mDecoderSummary.drmDataHeader);

  /** check orbit **/
  if (mCheckerVerbose) {
    printf(" --- Checking DRM/RDH orbit: %08x/%08x \n", orbit, mDecoderRDH->orbit);
  }
  if (orbit != mDecoderRDH->orbit) {
    if (mCheckerVerbose) {
      printf(" DRM/RDH orbit mismatch: %08x/%08x \n", orbit, mDecoderRDH->orbit);
    }
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_ORBIT_MISMATCH;
  }

  /** check FEE id **/
  if (mCheckerVerbose) {
    printf(" --- Checking DRM/RDH FEE id: %d/%d \n", drmId, mDecoderRDH->feeId & 0xFF);
  }
  if (drmId != (mDecoderRDH->feeId & 0xFF)) {
    if (mCheckerVerbose) {
      printf(" DRM/RDH FEE id mismatch: %d/%d \n", drmId, mDecoderRDH->feeId & 0xFF);
    }
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_FEEID_MISMATCH;
  }
}

template <>
void Compressor<o2::header::RAWDataHeaderV6, false>::checkerCheckRDH()
{
  uint32_t orbit = *mDecoderSummary.tofOrbit;
  uint32_t drmId = GET_DRMDATAHEADER_DRMID(*mDecoderSummary.drmDataHeader);

  /** check orbit **/
  if (orbit != mDecoderRDH->orbit) {
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_ORBIT_MISMATCH;
  }

  /** check FEE id **/
  if (drmId != (mDecoderRDH->feeId & 0xFF)) {
    mCheckerSummary.DiagnosticWord[0] |= diagnostic::DRM_FEEID_MISMATCH;
  }
}

template <typename RDH, bool verbose>
void Compressor<RDH, verbose>::resetCounters()
{
  mEventCounter = 0;
  mFatalCounter = 0;
  mErrorCounter = 0;
  mDRMCounters = {0};
  for (int itrm = 0; itrm < 10; ++itrm) {
    mTRMCounters[itrm] = {0};
    for (int ichain = 0; ichain < 2; ++ichain) {
      mTRMChainCounters[itrm][ichain] = {0};
    }
  }
}

template <typename RDH, bool verbose>
void Compressor<RDH, verbose>::checkSummary()
{
  char chname[2] = {'a', 'b'};

  std::cout << colorBlue
            << "--- SUMMARY COUNTERS: " << mEventCounter << " events "
            << " | " << mFatalCounter << " decode fatals "
            << " | " << mErrorCounter << " decode errors "
            << colorReset
            << std::endl;
#ifndef CHECKER_COUNTER
  return;
#endif
  if (mEventCounter == 0) {
    return;
  }
  printf("\n");
  printf("    DRM ");
  float drmheaders = 100. * (float)mDRMCounters.Headers / (float)mEventCounter;
  printf("  \033%sheaders: %5.1f %%\033[0m ", drmheaders < 100. ? "[1;31m" : "[0m", drmheaders);
  if (mDRMCounters.Headers == 0) {
    printf("\n");
    return;
  }
  float cbit = 100. * (float)mDRMCounters.clockStatus / float(mDRMCounters.Headers);
  printf("     \033%sCbit: %5.1f %%\033[0m ", cbit > 0. ? "[1;31m" : "[0m", cbit);
  float fault = 100. * (float)mDRMCounters.Fault / float(mDRMCounters.Headers);
  printf("    \033%sfault: %5.1f %%\033[0m ", fault > 0. ? "[1;31m" : "[0m", cbit);
  float rtobit = 100. * (float)mDRMCounters.RTOBit / float(mDRMCounters.Headers);
  printf("   \033%sRTObit: %5.1f %%\033[0m ", rtobit > 0. ? "[1;31m" : "[0m", cbit);
  printf("\n");
  //      std::cout << "-----------------------------------------------------------" << std::endl;
  //      printf("    LTM | headers: %5.1f %% \n", 0.);
  for (int itrm = 0; itrm < 10; ++itrm) {
    printf("\n");
    printf(" %2d TRM ", itrm + 3);
    float trmheaders = 100. * (float)mTRMCounters[itrm].Headers / float(mDRMCounters.Headers);
    printf("  \033%sheaders: %5.1f %%\033[0m ", trmheaders < 100. ? "[1;31m" : "[0m", trmheaders);
    if (mTRMCounters[itrm].Headers == 0.) {
      printf("\n");
      continue;
    }
    float empty = 100. * (float)mTRMCounters[itrm].Empty / (float)mTRMCounters[itrm].Headers;
    printf("    \033%sempty: %5.1f %%\033[0m ", empty > 0. ? "[1;31m" : "[0m", empty);
    float evCount = 100. * (float)mTRMCounters[itrm].EventCounterMismatch / (float)mTRMCounters[itrm].Headers;
    printf("  \033%sevCount: %5.1f %%\033[0m ", evCount > 0. ? "[1;31m" : "[0m", evCount);
    float ebit = 100. * (float)mTRMCounters[itrm].EBit / (float)mTRMCounters[itrm].Headers;
    printf("     \033%sEbit: %5.1f %%\033[0m ", ebit > 0. ? "[1;31m" : "[0m", ebit);
    printf(" \n");
    for (int ichain = 0; ichain < 2; ++ichain) {
      printf("      %c ", chname[ichain]);
      float chainheaders = 100. * (float)mTRMChainCounters[itrm][ichain].Headers / (float)mTRMCounters[itrm].Headers;
      printf("  \033%sheaders: %5.1f %%\033[0m ", chainheaders < 100. ? "[1;31m" : "[0m", chainheaders);
      if (mTRMChainCounters[itrm][ichain].Headers == 0) {
        printf("\n");
        continue;
      }
      float status = 100. * mTRMChainCounters[itrm][ichain].BadStatus / (float)mTRMChainCounters[itrm][ichain].Headers;
      printf("   \033%sstatus: %5.1f %%\033[0m ", status > 0. ? "[1;31m" : "[0m", status);
      float bcid = 100. * mTRMChainCounters[itrm][ichain].BunchIDMismatch / (float)mTRMChainCounters[itrm][ichain].Headers;
      printf("     \033%sbcID: %5.1f %%\033[0m ", bcid > 0. ? "[1;31m" : "[0m", bcid);
      float tdcerr = 100. * mTRMChainCounters[itrm][ichain].TDCerror / (float)mTRMChainCounters[itrm][ichain].Headers;
      printf("   \033%sTDCerr: %5.1f %%\033[0m ", tdcerr > 0. ? "[1;31m" : "[0m", tdcerr);
      printf("\n");
    }
  }
  printf("\n");
}

template class Compressor<o2::header::RAWDataHeaderV4, false>;
template class Compressor<o2::header::RAWDataHeaderV4, true>;
template class Compressor<o2::header::RAWDataHeaderV6, false>;
template class Compressor<o2::header::RAWDataHeaderV6, true>;

} // namespace tof
} // namespace o2
