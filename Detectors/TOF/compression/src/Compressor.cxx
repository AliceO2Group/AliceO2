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
#include "DetectorsRaw/HBFUtils.h"

#include <cstring>
#include <iostream>

#define DECODER_PARANOID
//#define DECODER_VERBOSE
//#define ENCODER_VERBOSE
//#define CHECKER_VERBOSE
//#define CHECKER_COUNTER

#ifdef DECODER_VERBOSE
#warning "Building code with DecoderVerbose option. This may limit the speed."
#endif
#ifdef ENCODER_VERBOSE
#warning "Building code with EncoderVerbose option. This may limit the speed."
#endif
#ifdef CHECKER_VERBOSE
#warning "Building code with CheckerVerbose option. This may limit the speed."
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
#define IS_TDC_ERROR(x) ((x & 0xF0000000) == 0x60000000)
#define IS_FILLER(x) ((x & 0xFFFFFFFF) == 0x70000000)
#define IS_TDC_HIT(x) ((x & 0x80000000) == 0x80000000)

// DRM getters
//#define GET_DRMDATAHEADER_DRMID(x) ((x & 0x0FE00000) >> 21)
#define GET_DRMDATAHEADER_DRMID(x) ((x >> 21) & 0x7F)
#define GET_DRMHEADW1_PARTSLOTMASK(x) ((x & 0x00007FF0) >> 4)
#define GET_DRMHEADW1_CLOCKSTATUS(x) ((x & 0x00018000) >> 15)
#define GET_DRMHEADW2_ENASLOTMASK(x) ((x & 0x00007FF0) >> 4)
#define GET_DRMHEADW2_FAULTSLOTMASK(x) ((x & 0x07FF0000) >> 16)
#define GET_DRMHEADW2_READOUTTIMEOUT(x) ((x & 0x08000000) >> 27)
#define GET_DRMHEADW3_GBTBUNCHCNT(x) ((x & 0x0000FFF0) >> 4)
#define GET_DRMDATATRAILER_LOCEVCNT(x) ((x & 0x0000FFF0) >> 4)

// TRM getter
#define GET_TRMDATAHEADER_SLOTID(x) ((x & 0x0000000F))
#define GET_TRMDATAHEADER_EVENTCNT(x) ((x & 0x07FE0000) >> 17)
#define GET_TRMDATAHEADER_EVENTWORDS(x) ((x & 0x0001FFF0) >> 4)
#define GET_TRMDATAHEADER_EMPTYBIT(x) ((x & 0x08000000) >> 27)

// TRM Chain getters
#define GET_TRMCHAINHEADER_SLOTID(x) ((x & 0x0000000F))
#define GET_TRMCHAINHEADER_BUNCHCNT(x) ((x & 0x0000FFF0) >> 4)
#define GET_TRMCHAINTRAILER_EVENTCNT(x) ((x & 0x0FFF0000) >> 16)
#define GET_TRMCHAINTRAILER_STATUS(x) ((x & 0x0000000F))

// TDC getters
#define GET_TRMDATAHIT_TIME(x) ((x & 0x001FFFFF))
#define GET_TRMDATAHIT_CHANID(x) ((x & 0x00E00000) >> 21)
#define GET_TRMDATAHIT_TDCID(x) ((x & 0x0F000000) >> 24)
#define GET_TRMDATAHIT_EBIT(x) ((x & 0x10000000) >> 28)
#define GET_TRMDATAHIT_PSBITS(x) ((x & 0x60000000) >> 29)

namespace o2
{
namespace tof
{

Compressor::~Compressor()
{
  if (mDecoderBuffer && mOwnDecoderBuffer)
    delete[] mDecoderBuffer;
  if (mEncoderBuffer && mOwnDecoderBuffer)
    delete[] mEncoderBuffer;
}

bool Compressor::init()
{
  if (decoderInit())
    return true;
  if (encoderInit())
    return true;
  return false;
}

bool Compressor::open(const std::string inFileName, const std::string outFileName)
{
  if (decoderOpen(inFileName))
    return true;
  if (encoderOpen(outFileName))
    return true;
  return false;
}

bool Compressor::close()
{
  if (decoderClose())
    return true;
  if (encoderClose())
    return true;
  return false;
}

bool Compressor::decoderInit()
{
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    std::cout << colorBlue
              << "--- INITIALISE DECODER BUFFER: " << mDecoderBufferSize << " bytes"
              << colorReset
              << std::endl;
  }
#endif
  if (mDecoderBuffer && mOwnDecoderBuffer) {
    std::cout << colorYellow
              << "--- a buffer was already allocated, cleaning"
              << colorReset
              << std::endl;
    delete[] mDecoderBuffer;
  }
  mDecoderBuffer = new char[mDecoderBufferSize];
  mOwnDecoderBuffer = true;
  return false;
}

bool Compressor::encoderInit()
{
#ifdef ENCODER_VERBOSE
  if (mEncoderVerbose) {
    std::cout << colorBlue
              << "--- INITIALISE ENCODER BUFFER: " << mEncoderBufferSize << " bytes"
              << colorReset
              << std::endl;
  }
#endif
  if (mEncoderBuffer && mOwnEncoderBuffer) {
    std::cout << colorYellow
              << "-W- a buffer was already allocated, cleaning"
              << colorReset
              << std::endl;
    delete[] mEncoderBuffer;
  }
  mEncoderBuffer = new char[mEncoderBufferSize];
  mOwnEncoderBuffer = true;
  encoderRewind();
  return false;
}

bool Compressor::decoderOpen(const std::string name)
{
  if (mDecoderFile.is_open()) {
    std::cout << colorYellow
              << "-W- a file was already open, closing"
              << colorReset
              << std::endl;
    mDecoderFile.close();
  }
  mDecoderFile.open(name.c_str(), std::fstream::in | std::fstream::binary);
  if (!mDecoderFile.is_open()) {
    std::cerr << colorRed
              << "-E- Cannot open input file: " << name
              << colorReset
              << std::endl;
    return true;
  }
  return false;
}

bool Compressor::encoderOpen(const std::string name)
{
  if (mEncoderFile.is_open()) {
    std::cout << colorYellow
              << "-W- a file was already open, closing"
              << colorReset
              << std::endl;
    mEncoderFile.close();
  }
  mEncoderFile.open(name.c_str(), std::fstream::out | std::fstream::binary);
  if (!mEncoderFile.is_open()) {
    std::cerr << colorRed << "-E- Cannot open output file: " << name
              << colorReset
              << std::endl;
    return true;
  }
  return false;
}

bool Compressor::decoderClose()
{
  if (mDecoderFile.is_open()) {
    mDecoderFile.close();
    return false;
  }
  return true;
}

bool Compressor::encoderClose()
{
  if (mEncoderFile.is_open())
    mEncoderFile.close();
  return false;
}

bool Compressor::decoderRead()
{
  if (!mDecoderFile.is_open()) {
    std::cout << colorRed << "--- no input file is open"
              << colorReset
              << std::endl;
    return true;
  }

  char* inputPointer = mDecoderBuffer;
  mDecoderFile.read(inputPointer, 64);
  mDecoderBufferSize = 64;
  auto rdh = reinterpret_cast<o2::header::RAWDataHeader*>(inputPointer);
  while (!rdh->stop) {
    mDecoderFile.read(inputPointer + rdh->headerSize, rdh->offsetToNext - rdh->headerSize);
    mDecoderBufferSize += (rdh->offsetToNext - rdh->headerSize);
    inputPointer += rdh->offsetToNext;
    mDecoderFile.read(inputPointer, 64);
    mDecoderBufferSize += 64;
    rdh = reinterpret_cast<o2::header::RAWDataHeader*>(inputPointer);
  }

  /** check end of file **/
  if (mDecoderFile.eof()) {
    std::cout << colorRed << "--- Nothing else to read"
              << colorReset
              << std::endl;
    return true;
  }

#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    std::cout << colorBlue
              << "--- DECODER READ HBF: " << mDecoderBufferSize << " bytes"
              << colorReset
              << std::endl;
  }
#endif

  return false;
}

bool Compressor::encoderWrite()
{
#ifdef ENCODER_VERBOSE
  if (mEncoderVerbose) {
    std::cout << colorBlue
              << "--- ENCODER WRITE BUFFER: " << getEncoderByteCounter() << " bytes"
              << colorReset
              << std::endl;
  }
#endif
  mEncoderFile.write(mEncoderBuffer, getEncoderByteCounter());
  encoderRewind();
  return false;
}

bool Compressor::decoderParanoid()
{
  if (mDecoderPointer >= mDecoderPointerMax) {
    printf("%s %08x [ERROR] fatal error: beyond memory size %s \n", colorRed, *mDecoderPointer, colorReset);
    mDecoderFatal = true;
    return true;
  }
  return false;
}

bool Compressor::processHBF()
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
  mEncoderRDH = reinterpret_cast<o2::header::RAWDataHeader*>(mEncoderPointer);
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

  /** copy RDH open to encoder buffer **/
  std::memcpy(mEncoderPointer, mDecoderRDH, mDecoderRDH->headerSize);
  mEncoderPointer = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(mEncoderPointer) + rdh->headerSize);

  /** process DRM data **/
  mDecoderPointer = reinterpret_cast<uint32_t*>(mDecoderSaveBuffer);
  mDecoderPointerMax = reinterpret_cast<uint32_t*>(mDecoderSaveBuffer + mDecoderSaveBufferDataSize);
  while (mDecoderPointer < mDecoderPointerMax) {
    mEventCounter++;
    if (processDRM())
      break;
  }
  mDecoderSaveBufferDataSize = 0;

  /** bring encoder pointer back if fatal error **/
  if (mDecoderFatal) {
    mFatalCounter++;
    mEncoderPointer = mEncoderPointerStart;
  }

  if (mDecoderError)
    mErrorCounter++;

  /** updated encoder RDH open **/
  mEncoderRDH->memorySize = reinterpret_cast<char*>(mEncoderPointer) - reinterpret_cast<char*>(mEncoderRDH);
  mEncoderRDH->offsetToNext = mEncoderRDH->memorySize;

  /** copy RDH close to encoder buffer **/
  /** CAREFUL WITH THE PAGE COUNTER **/
  mEncoderRDH = reinterpret_cast<o2::header::RAWDataHeader*>(mEncoderPointer);
  std::memcpy(mEncoderRDH, rdh, rdh->headerSize);
  mEncoderRDH->memorySize = rdh->headerSize;
  mEncoderRDH->offsetToNext = mEncoderRDH->memorySize;
  mEncoderPointer = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(mEncoderPointer) + rdh->headerSize);

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

bool Compressor::processDRM()
{

#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    std::cout << colorBlue << "--- PROCESS DRM"
              << colorReset
              << std::endl;
  }
#endif

  /** init decoder **/
  mDecoderNextWord = 1;
  mDecoderError = false;
  mDecoderFatal = false;
  mEncoderPointerStart = mEncoderPointer;

  /** check TOF Data Header **/
  if (!IS_DRM_COMMON_HEADER(*mDecoderPointer)) {
#ifdef DECODER_VERBOSE
    printf("%s %08x [ERROR] fatal error %s \n", colorRed, *mDecoderPointer, colorReset);
#endif
    mDecoderFatal = true;
    return true;
  }
  mDecoderSummary.tofDataHeader = *mDecoderPointer;
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    auto tofDataHeader = reinterpret_cast<raw::TOFDataHeader_t*>(mDecoderPointer);
    auto bytePayload = tofDataHeader->bytePayload;
    printf(" %08x TOF Data Header       (bytePayload=%d) \n", *mDecoderPointer, bytePayload);
  }
#endif
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** TOF Orbit **/
  mDecoderSummary.tofOrbit = *mDecoderPointer;
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    auto tofOrbit = reinterpret_cast<raw::TOFOrbit_t*>(mDecoderPointer);
    auto orbit = tofOrbit->orbit;
    printf(" %08x TOF Orbit             (orbit=%d) \n", *mDecoderPointer, orbit);
  }
#endif
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** check DRM Data Header **/
  if (!IS_DRM_GLOBAL_HEADER(*mDecoderPointer)) {
#ifdef DECODER_VERBOSE
    printf("%s %08x [ERROR] fatal error %s \n", colorRed, *mDecoderPointer, colorReset);
#endif
    mDecoderFatal = true;
    return true;
  }
  mDecoderSummary.drmDataHeader = *mDecoderPointer;
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    auto drmDataHeader = reinterpret_cast<raw::DRMDataHeader_t*>(mDecoderPointer);
    auto drmId = drmDataHeader->drmId;
    printf(" %08x DRM Data Header       (drmId=%d) \n", *mDecoderPointer, drmId);
  }
#endif
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** DRM Header Word 1 **/
  mDecoderSummary.drmHeadW1 = *mDecoderPointer;
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    auto drmHeadW1 = reinterpret_cast<raw::DRMHeadW1_t*>(mDecoderPointer);
    auto partSlotMask = drmHeadW1->partSlotMask;
    auto clockStatus = drmHeadW1->clockStatus;
    auto drmHSize = drmHeadW1->drmHSize;
    printf(" %08x DRM Header Word 1     (partSlotMask=0x%03x, clockStatus=%d, drmHSize=%d) \n", *mDecoderPointer, partSlotMask, clockStatus, drmHSize);
  }
#endif
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** DRM Header Word 2 **/
  mDecoderSummary.drmHeadW2 = *mDecoderPointer;
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    auto drmHeadW2 = reinterpret_cast<raw::DRMHeadW2_t*>(mDecoderPointer);
    auto enaSlotMask = drmHeadW2->enaSlotMask;
    auto faultSlotMask = drmHeadW2->faultSlotMask;
    auto readoutTimeOut = drmHeadW2->readoutTimeOut;
    printf(" %08x DRM Header Word 2     (enaSlotMask=0x%03x, faultSlotMask=%d, readoutTimeOut=%d) \n", *mDecoderPointer, enaSlotMask, faultSlotMask, readoutTimeOut);
  }
#endif
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** DRM Header Word 3 **/
  mDecoderSummary.drmHeadW3 = *mDecoderPointer;
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    auto drmHeadW3 = reinterpret_cast<raw::DRMHeadW3_t*>(mDecoderPointer);
    auto gbtBunchCnt = drmHeadW3->gbtBunchCnt;
    auto locBunchCnt = drmHeadW3->locBunchCnt;
    printf(" %08x DRM Header Word 3     (gbtBunchCnt=%d, locBunchCnt=%d) \n", *mDecoderPointer, gbtBunchCnt, locBunchCnt);
  }
#endif
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** DRM Header Word 4 **/
  mDecoderSummary.drmHeadW4 = *mDecoderPointer;
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    printf(" %08x DRM Header Word 4   \n", *mDecoderPointer);
  }
#endif
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** DRM Header Word 5 **/
  mDecoderSummary.drmHeadW5 = *mDecoderPointer;
#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    printf(" %08x DRM Header Word 5   \n", *mDecoderPointer);
  }
#endif
#ifdef DECODER_PARANOID
  if (decoderParanoid())
    return true;
#endif
  decoderNext();

  /** encode Crate Header **/
  *mEncoderPointer = 0x80000000;
  *mEncoderPointer |= GET_DRMHEADW2_ENASLOTMASK(mDecoderSummary.drmHeadW2) << 12;
  *mEncoderPointer |= GET_DRMDATAHEADER_DRMID(mDecoderSummary.drmDataHeader) << 24;
  *mEncoderPointer |= GET_DRMHEADW3_GBTBUNCHCNT(mDecoderSummary.drmHeadW3);
#ifdef ENCODER_VERBOSE
  if (mEncoderVerbose) {
    auto crateHeader = reinterpret_cast<compressed::CrateHeader_t*>(mEncoderPointer);
    auto bunchID = crateHeader->bunchID;
    auto drmID = crateHeader->drmID;
    auto slotEnableMask = crateHeader->slotEnableMask;
    printf("%s %08x Crate header          (drmID=%d, bunchID=%d, slotEnableMask=0x%x) %s \n", colorGreen, *mEncoderPointer, drmID, bunchID, slotEnableMask, colorReset);
  }
#endif
  encoderNext();

  /** encode Crate Orbit **/
  *mEncoderPointer = mDecoderSummary.tofOrbit;
#ifdef ENCODER_VERBOSE
  if (mEncoderVerbose) {
    auto crateOrbit = reinterpret_cast<compressed::CrateOrbit_t*>(mEncoderPointer);
    auto orbitID = crateOrbit->orbitID;
    printf("%s %08x Crate orbit           (orbitID=%d) %s \n", colorGreen, *mEncoderPointer, orbitID, colorReset);
  }
#endif
  encoderNext();

  /** loop over DRM payload **/
  while (true) {
    /** LTM global header detected **/
    if (IS_LTM_GLOBAL_HEADER(*mDecoderPointer)) {

#ifdef DECODER_VERBOSE
      if (mDecoderVerbose) {
        printf(" %08x LTM Global Header \n", *mDecoderPointer);
      }
#endif
#ifdef DECODER_PARANOID
      if (decoderParanoid())
        return true;
#endif
      decoderNext();

      /** loop over LTM payload **/
      while (true) {
        /** LTM global trailer detected **/
        if (IS_LTM_GLOBAL_TRAILER(*mDecoderPointer)) {
#ifdef DECODER_VERBOSE
          if (mDecoderVerbose) {
            printf(" %08x LTM Global Trailer \n", *mDecoderPointer);
          }
#endif
#ifdef DECODER_PARANOID
          if (decoderParanoid())
            return true;
#endif
          decoderNext();
          break;
        }

#ifdef DECODER_VERBOSE
        if (mDecoderVerbose) {
          printf(" %08x LTM data \n", *mDecoderPointer);
        }
#endif
#ifdef DECODER_PARANOID
        if (decoderParanoid())
          return true;
#endif
        decoderNext();
      }
    }

    /** TRM Data Header detected **/
    if (IS_TRM_GLOBAL_HEADER(*mDecoderPointer) && GET_TRMDATAHEADER_SLOTID(*mDecoderPointer) > 2) {
      uint32_t slotId = GET_TRMDATAHEADER_SLOTID(*mDecoderPointer);
      int itrm = slotId - 3;
      mDecoderSummary.trmDataHeader[itrm] = *mDecoderPointer;
#ifdef DECODER_VERBOSE
      if (mDecoderVerbose) {
        auto trmDataHeader = reinterpret_cast<raw::TRMDataHeader_t*>(mDecoderPointer);
        auto eventWords = trmDataHeader->eventWords;
        auto eventCnt = trmDataHeader->eventCnt;
        auto emptyBit = trmDataHeader->emptyBit;
        printf(" %08x TRM Data Header       (slotId=%u, eventWords=%d, eventCnt=%d, emptyBit=%01x) \n", *mDecoderPointer, slotId, eventWords, eventCnt, emptyBit);
      }
#endif
#ifdef DECODER_PARANOID
      if (decoderParanoid())
        return true;
#endif
      decoderNext();

      /** loop over TRM payload **/
      while (true) {
        /** TRM Chain-A Header detected **/
        if (IS_TRM_CHAINA_HEADER(*mDecoderPointer) && GET_TRMCHAINHEADER_SLOTID(*mDecoderPointer) == slotId) {
          mDecoderSummary.trmChainHeader[itrm][0] = *mDecoderPointer;
          mDecoderSummary.hasHits[itrm][0] = false;
          mDecoderSummary.hasErrors[itrm][0] = false;
#ifdef DECODER_VERBOSE
          if (mDecoderVerbose) {
            auto trmChainHeader = reinterpret_cast<raw::TRMChainHeader_t*>(mDecoderPointer);
            auto bunchCnt = trmChainHeader->bunchCnt;
            printf(" %08x TRM Chain-A Header    (slotId=%u, bunchCnt=%d) \n", *mDecoderPointer, slotId, bunchCnt);
          }
#endif
#ifdef DECODER_PARANOID
          if (decoderParanoid())
            return true;
#endif
          decoderNext();

          /** loop over TRM Chain-A payload **/
          while (true) {
            /** TDC hit detected **/
            if (IS_TDC_HIT(*mDecoderPointer)) {
              mDecoderSummary.hasHits[itrm][0] = true;
              auto itdc = GET_TRMDATAHIT_TDCID(*mDecoderPointer);
              auto ihit = mDecoderSummary.trmDataHits[0][itdc];
              mDecoderSummary.trmDataHit[0][itdc][ihit] = *mDecoderPointer;
              mDecoderSummary.trmDataHits[0][itdc]++;
#ifdef DECODER_VERBOSE
              if (mDecoderVerbose) {
                auto trmDataHit = reinterpret_cast<raw::TRMDataHit_t*>(mDecoderPointer);
                auto time = trmDataHit->time;
                auto chanId = trmDataHit->chanId;
                auto tdcId = trmDataHit->tdcId;
                auto dataId = trmDataHit->dataId;
                printf(" %08x TRM Data Hit          (time=%d, chanId=%d, tdcId=%d, dataId=0x%x) \n", *mDecoderPointer, time, chanId, tdcId, dataId);
              }
#endif
#ifdef DECODER_PARANOID
              if (decoderParanoid())
                return true;
#endif
              decoderNext();
              continue;
            }

            /** TDC error detected **/
            if (IS_TDC_ERROR(*mDecoderPointer)) {
              mDecoderSummary.hasErrors[itrm][0] = true;
#ifdef DECODER_VERBOSE
              if (mDecoderVerbose) {
                printf("%s %08x TDC error %s \n", colorRed, *mDecoderPointer, colorReset);
              }
#endif
#ifdef DECODER_PARANOID
              if (decoderParanoid())
                return true;
#endif
              decoderNext();
              continue;
            }

            /** TRM Chain-A Trailer detected **/
            if (IS_TRM_CHAINA_TRAILER(*mDecoderPointer)) {
              mDecoderSummary.trmChainTrailer[itrm][0] = *mDecoderPointer;
#ifdef DECODER_VERBOSE
              if (mDecoderVerbose) {
                auto trmChainTrailer = reinterpret_cast<raw::TRMChainTrailer_t*>(mDecoderPointer);
                auto eventCnt = trmChainTrailer->eventCnt;
                printf(" %08x TRM Chain-A Trailer   (slotId=%u, eventCnt=%d) \n", *mDecoderPointer, slotId, eventCnt);
              }
#endif
#ifdef DECODER_PARANOID
              if (decoderParanoid())
                return true;
#endif
              decoderNext();
              break;
            }

            /** decode error **/
            mDecoderError = true;
#ifdef DECODER_VERBOSE
            if (mDecoderVerbose) {
              printf("%s %08x [ERROR] breaking TRM Chain-A decode stream %s \n", colorRed, *mDecoderPointer, colorReset);
            }
#endif
            /** decode error detected, be paranoid **/
            if (decoderParanoid())
              return true;

            decoderNext();
            break;
          }
        } /** end of loop over TRM chain-A payload **/

        /** TRM Chain-B Header detected **/
        if (IS_TRM_CHAINB_HEADER(*mDecoderPointer) && GET_TRMCHAINHEADER_SLOTID(*mDecoderPointer) == slotId) {
          mDecoderSummary.hasHits[itrm][1] = false;
          mDecoderSummary.hasErrors[itrm][1] = false;
          mDecoderSummary.trmChainHeader[itrm][1] = *mDecoderPointer;
#ifdef DECODER_VERBOSE
          if (mDecoderVerbose) {
            auto trmChainHeader = reinterpret_cast<raw::TRMChainHeader_t*>(mDecoderPointer);
            auto bunchCnt = trmChainHeader->bunchCnt;
            printf(" %08x TRM Chain-B Header    (slotId=%u, bunchCnt=%d) \n", *mDecoderPointer, slotId, bunchCnt);
          }
#endif
#ifdef DECODER_PARANOID
          if (decoderParanoid())
            return true;
#endif
          decoderNext();

          /** loop over TRM Chain-B payload **/
          while (true) {
            /** TDC hit detected **/
            if (IS_TDC_HIT(*mDecoderPointer)) {
              mDecoderSummary.hasHits[itrm][1] = true;
              auto itdc = GET_TRMDATAHIT_TDCID(*mDecoderPointer);
              auto ihit = mDecoderSummary.trmDataHits[1][itdc];
              mDecoderSummary.trmDataHit[1][itdc][ihit] = *mDecoderPointer;
              mDecoderSummary.trmDataHits[1][itdc]++;
#ifdef DECODER_VERBOSE
              if (mDecoderVerbose) {
                auto trmDataHit = reinterpret_cast<raw::TRMDataHit_t*>(mDecoderPointer);
                auto time = trmDataHit->time;
                auto chanId = trmDataHit->chanId;
                auto tdcId = trmDataHit->tdcId;
                auto dataId = trmDataHit->dataId;
                printf(" %08x TRM Data Hit          (time=%d, chanId=%d, tdcId=%d, dataId=0x%x \n", *mDecoderPointer, time, chanId, tdcId, dataId);
              }
#endif
#ifdef DECODER_PARANOID
              if (decoderParanoid())
                return true;
#endif
              decoderNext();
              continue;
            }

            /** TDC error detected **/
            if (IS_TDC_ERROR(*mDecoderPointer)) {
              mDecoderSummary.hasErrors[itrm][1] = true;
#ifdef DECODER_VERBOSE
              if (mDecoderVerbose) {
                printf("%s %08x TDC error %s \n", colorRed, *mDecoderPointer, colorReset);
              }
#endif
#ifdef DECODER_PARANOID
              if (decoderParanoid())
                return true;
#endif
              decoderNext();
              continue;
            }

            /** TRM Chain-B trailer detected **/
            if (IS_TRM_CHAINB_TRAILER(*mDecoderPointer)) {
              mDecoderSummary.trmChainTrailer[itrm][1] = *mDecoderPointer;
#ifdef DECODER_VERBOSE
              if (mDecoderVerbose) {
                auto trmChainTrailer = reinterpret_cast<raw::TRMChainTrailer_t*>(mDecoderPointer);
                auto eventCnt = trmChainTrailer->eventCnt;
                printf(" %08x TRM Chain-B Trailer   (slotId=%u, eventCnt=%d) \n", *mDecoderPointer, slotId, eventCnt);
              }
#endif
#ifdef DECODER_PARANOID
              if (decoderParanoid())
                return true;
#endif
              decoderNext();
              break;
            }

            /** decode error **/
            mDecoderError = true;
#ifdef DECODER_VERBOSE
            if (mDecoderVerbose) {
              printf("%s %08x [ERROR] breaking TRM Chain-B decode stream %s \n", colorRed, *mDecoderPointer, colorReset);
            }
#endif
            /** decode error detected, be paranoid **/
            if (decoderParanoid())
              return true;

            decoderNext();
            break;
          }
        } /** end of loop over TRM chain-A payload **/

        /** TRM Data Trailer detected **/
        if (IS_TRM_GLOBAL_TRAILER(*mDecoderPointer)) {
          mDecoderSummary.trmDataTrailer[itrm] = *mDecoderPointer;
#ifdef DECODER_VERBOSE
          if (mDecoderVerbose) {
            auto trmDataTrailer = reinterpret_cast<raw::TRMDataTrailer_t*>(mDecoderPointer);
            auto eventCRC = trmDataTrailer->eventCRC;
            auto lutErrorBit = trmDataTrailer->lutErrorBit;
            printf(" %08x TRM Data Trailer      (slotId=%u, eventCRC=%d, lutErrorBit=%d) \n", *mDecoderPointer, slotId, eventCRC, lutErrorBit);
          }
#endif
#ifdef DECODER_PARANOID
          if (decoderParanoid())
            return true;
#endif
          decoderNext();

          /** encoder Spider **/
          if (mDecoderSummary.hasHits[itrm][0] || mDecoderSummary.hasHits[itrm][1])
            encoderSpider(itrm);

          /** filler detected **/
          if (IS_FILLER(*mDecoderPointer)) {
#ifdef DECODER_VERBOSE
            if (mDecoderVerbose) {
              printf(" %08x Filler \n", *mDecoderPointer);
            }
#endif
#ifdef DECODER_PARANOID
            if (decoderParanoid())
              return true;
#endif
            decoderNext();
          }

          break;
        }

        /** decode error **/
        mDecoderError = true;
#ifdef DECODER_VERBOSE
        if (mDecoderVerbose) {
          printf("%s %08x [ERROR] breaking TRM decode stream %s \n", colorRed, *mDecoderPointer, colorReset);
        }
#endif
        /** decode error detected, be paranoid **/
        if (decoderParanoid())
          return true;

        decoderNext();
        break;

      } /** end of loop over TRM payload **/

      continue;
    }

    /** DRM Data Trailer detected **/
    if (IS_DRM_GLOBAL_TRAILER(*mDecoderPointer)) {
      mDecoderSummary.drmDataTrailer = *mDecoderPointer;
#ifdef DECODER_VERBOSE
      if (mDecoderVerbose) {
        auto drmDataTrailer = reinterpret_cast<raw::DRMDataTrailer_t*>(mDecoderPointer);
        auto locEvCnt = drmDataTrailer->locEvCnt;
        printf(" %08x DRM Data Trailer      (locEvCnt=%d) \n", *mDecoderPointer, locEvCnt);
      }
#endif
#ifdef DECODER_PARANOID
      if (decoderParanoid())
        return true;
#endif
      decoderNext();

      /** filler detected **/
      if (IS_FILLER(*mDecoderPointer)) {
#ifdef DECODER_VERBOSE
        if (mDecoderVerbose) {
          printf(" %08x Filler \n", *mDecoderPointer);
        }
#endif
#ifdef DECODER_PARANOID
        if (decoderParanoid())
          return true;
#endif
        decoderNext();
      }

      /** check event **/
      checkerCheck();

      /** encode Crate Trailer **/
      *mEncoderPointer = 0x80000000;
      *mEncoderPointer |= mCheckerSummary.nDiagnosticWords;
      *mEncoderPointer |= GET_DRMDATATRAILER_LOCEVCNT(mDecoderSummary.drmDataTrailer) << 4;
#ifdef ENCODER_VERBOSE
      if (mEncoderVerbose) {
        auto CrateTrailer = reinterpret_cast<compressed::CrateTrailer_t*>(mEncoderPointer);
        auto EventCounter = CrateTrailer->eventCounter;
        auto NumberOfDiagnostics = CrateTrailer->numberOfDiagnostics;
        printf("%s %08x Crate trailer         (EventCounter=%d, NumberOfDiagnostics=%d) %s \n", colorGreen, *mEncoderPointer, EventCounter, NumberOfDiagnostics, colorReset);
      }
#endif
      encoderNext();

      /** encode Diagnostic Words **/
      for (int iword = 0; iword < mCheckerSummary.nDiagnosticWords; ++iword) {
        *mEncoderPointer = mCheckerSummary.DiagnosticWord[iword];
#ifdef ENCODER_VERBOSE
        if (mEncoderVerbose) {
          auto Diagnostic = reinterpret_cast<compressed::Diagnostic_t*>(mEncoderPointer);
          auto slotId = Diagnostic->slotID;
          auto FaultBits = Diagnostic->faultBits;
          printf("%s %08x Diagnostic            (slotId=%d, FaultBits=0x%x) %s \n", colorGreen, *mEncoderPointer, slotId, FaultBits, colorReset);
        }
#endif
        encoderNext();
      }

      mCheckerSummary.nDiagnosticWords = 0;

      break;
    }

#ifdef DECODER_VERBOSE
    if (mDecoderVerbose) {
      printf("%s %08x [ERROR] trying to recover DRM decode stream %s \n", colorRed, *mDecoderPointer, colorReset);
    }
#endif

    /** decode error detected, be paranoid **/
    if (decoderParanoid())
      return true;

    decoderNext();

  } /** end of loop over DRM payload **/

  mIntegratedBytes += getDecoderByteCounter();

#ifdef DECODER_VERBOSE
  if (mDecoderVerbose) {
    std::cout << colorBlue
              << "--- END PROCESS DRM"
              << colorReset
              << std::endl;
  }
#endif

  return false;
}

void Compressor::encoderSpider(int itrm)
{
  int slotId = itrm + 3;

  /** reset packed hits counter **/
  int firstFilledFrame = 255;
  int lastFilledFrame = 0;

  /** loop over TRM chains **/
  for (int ichain = 0; ichain < 2; ++ichain) {

    if (!mDecoderSummary.hasHits[itrm][ichain])
      continue;

    /** loop over TDCs **/
    for (int itdc = 0; itdc < 15; ++itdc) {

      auto nhits = mDecoderSummary.trmDataHits[ichain][itdc];
      if (nhits == 0)
        continue;

      /** loop over hits **/
      for (int ihit = 0; ihit < nhits; ++ihit) {

        auto lhit = mDecoderSummary.trmDataHit[ichain][itdc][ihit];
        if (GET_TRMDATAHIT_PSBITS(lhit) != 0x1) // must be a leading hit
          continue;

        auto chan = GET_TRMDATAHIT_CHANID(lhit);
        auto hitTime = GET_TRMDATAHIT_TIME(lhit);
        auto eBit = GET_TRMDATAHIT_EBIT(lhit);
        uint32_t totWidth = 0;

        // check next hits for packing
        for (int jhit = ihit + 1; jhit < nhits; ++jhit) {
          auto thit = mDecoderSummary.trmDataHit[ichain][itdc][jhit];
          if (GET_TRMDATAHIT_PSBITS(thit) == 0x2 && GET_TRMDATAHIT_CHANID(thit) == chan) { // must be a trailing hit from same channel
            totWidth = (GET_TRMDATAHIT_TIME(thit) - hitTime) / Geo::RATIO_TOT_TDC_BIN;     // compute TOT
            lhit = 0x0;                                                                    // mark as used
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

        if (iframe < firstFilledFrame)
          firstFilledFrame = iframe;
        if (iframe > lastFilledFrame)
          lastFilledFrame = iframe;
      }

      mDecoderSummary.trmDataHits[ichain][itdc] = 0;
    }
  }

  /** loop over frames **/
  for (int iframe = firstFilledFrame; iframe < lastFilledFrame + 1; iframe++) {

    /** check if frame is empty **/
    if (mSpiderSummary.nFramePackedHits[iframe] == 0)
      continue;

    // encode Frame Header
    *mEncoderPointer = 0x00000000;
    *mEncoderPointer |= slotId << 24;
    *mEncoderPointer |= iframe << 16;
    *mEncoderPointer |= mSpiderSummary.nFramePackedHits[iframe];
#ifdef ENCODER_VERBOSE
    if (mEncoderVerbose) {
      auto FrameHeader = reinterpret_cast<compressed::FrameHeader_t*>(mEncoderPointer);
      auto NumberOfHits = FrameHeader->numberOfHits;
      auto FrameID = FrameHeader->frameID;
      auto TRMID = FrameHeader->trmID;
      printf("%s %08x Frame header          (TRMID=%d, FrameID=%d, NumberOfHits=%d) %s \n", colorGreen, *mEncoderPointer, TRMID, FrameID, NumberOfHits, colorReset);
    }
#endif
    encoderNext();

    // packed hits
    for (int ihit = 0; ihit < mSpiderSummary.nFramePackedHits[iframe]; ++ihit) {
      *mEncoderPointer = mSpiderSummary.FramePackedHit[iframe][ihit];
#ifdef ENCODER_VERBOSE
      if (mEncoderVerbose) {
        auto PackedHit = reinterpret_cast<compressed::PackedHit_t*>(mEncoderPointer);
        auto Chain = PackedHit->chain;
        auto TDCID = PackedHit->tdcID;
        auto Channel = PackedHit->channel;
        auto Time = PackedHit->time;
        auto TOT = PackedHit->tot;
        printf("%s %08x Packed hit            (Chain=%d, TDCID=%d, Channel=%d, Time=%d, TOT=%d) %s \n", colorGreen, *mEncoderPointer, Chain, TDCID, Channel, Time, TOT, colorReset);
      }
#endif
      encoderNext();
    }

    mSpiderSummary.nFramePackedHits[iframe] = 0;
  }
}

bool Compressor::checkerCheck()
{
  mCheckerSummary.nDiagnosticWords = 0;
  mCheckerSummary.DiagnosticWord[0] = 0x00000001;

#ifdef CHECKER_VERBOSE
  if (mCheckerVerbose) {
    std::cout << colorBlue
              << "--- CHECK EVENT"
              << colorReset
              << std::endl;
  }
#endif

  /** increment check counter **/
  //    mCheckerCounter++;

  /** check TOF Data Header **/

  /** check DRM Data Header **/
  if (!mDecoderSummary.drmDataHeader) {
    mCheckerSummary.DiagnosticWord[0] |= DIAGNOSTIC_DRM_HEADER_MISSING;
#ifdef CHECKER_COUNTER
    mCheckerSummary.nDiagnosticWords++;
#endif
#ifdef CHECKER_VERBOSE
    if (mCheckerVerbose) {
      printf(" Missing DRM Data Header \n");
    }
#endif
    mDecoderSummary.tofDataHeader = 0x0;
    return true;
  }

  /** check DRM Data Trailer **/
  if (!mDecoderSummary.drmDataTrailer) {
    mCheckerSummary.DiagnosticWord[0] |= DIAGNOSTIC_DRM_TRAILER_MISSING;
#ifdef CHECKER_COUNTER
    mCheckerSummary.nDiagnosticWords++;
#endif
#ifdef CHECKER_VERBOSE
    if (mCheckerVerbose) {
      printf(" Missing DRM Data Trailer \n");
    }
#endif
    mDecoderSummary.tofDataHeader = 0x0;
    mDecoderSummary.drmDataHeader = 0x0;
    mDecoderSummary.drmDataTrailer = 0x0;
    return true;
  }

  /** increment DRM header counter **/
#ifdef CHECKER_COUNTER
  mDRMCounters.Headers++;
#endif

  /** get DRM relevant data **/
  uint32_t partSlotMask = GET_DRMHEADW1_PARTSLOTMASK(mDecoderSummary.drmHeadW1);
  uint32_t enaSlotMask = GET_DRMHEADW2_ENASLOTMASK(mDecoderSummary.drmHeadW2);
  uint32_t gbtBunchCnt = GET_DRMHEADW3_GBTBUNCHCNT(mDecoderSummary.drmHeadW3);
  uint32_t locEvCnt = GET_DRMDATATRAILER_LOCEVCNT(mDecoderSummary.drmDataTrailer);

  if (partSlotMask != enaSlotMask) {
#ifdef CHECKER_VERBOSE
    if (mCheckerVerbose) {
      printf(" Warning: enable/participating mask differ: %03x/%03x \n", enaSlotMask, partSlotMask);
    }
#endif
    mCheckerSummary.DiagnosticWord[0] |= DIAGNOSTIC_DRM_ENAPARTMASK_DIFFER;
  }

  /** check DRM clock status **/
  if (GET_DRMHEADW1_CLOCKSTATUS(mDecoderSummary.drmHeadW1) != 2) {
    mCheckerSummary.DiagnosticWord[0] |= DIAGNOSTIC_DRM_CLOCKSTATUS_WRONG;
#ifdef CHECKER_COUNTER
    mDRMCounters.clockStatus++;
#endif
#ifdef CHECKER_VERBOSE
    if (mCheckerVerbose) {
      printf("%s DRM wrong clock status: %d %s\n", colorRed, GET_DRMHEADW1_CLOCKSTATUS(mDecoderSummary.drmHeadW1), colorReset);
    }
#endif
  }

  /** check DRM fault mask **/
  if (GET_DRMHEADW2_FAULTSLOTMASK(mDecoderSummary.drmHeadW2)) {
    mCheckerSummary.DiagnosticWord[0] |= DIAGNOSTIC_DRM_FAULTSLOTMASK_NOTZERO;
#ifdef CHECKER_COUNTER
    mDRMCounters.Fault++;
#endif
#ifdef CHECKER_VERBOSE
    if (mCheckerVerbose) {
      printf(" DRM fault slot mask: %x \n", GET_DRMHEADW2_FAULTSLOTMASK(mDecoderSummary.drmHeadW2));
    }
#endif
  }

  /** check DRM readout timeout **/
  if (GET_DRMHEADW2_READOUTTIMEOUT(mDecoderSummary.drmHeadW2)) {
    mCheckerSummary.DiagnosticWord[0] |= DIAGNOSTIC_DRM_READOUTTIMEOUT_NOTZERO;
#ifdef CHECKER_COUNTER
    mDRMCounters.RTOBit++;
#endif
#ifdef CHECKER_VERBOSE
    if (mCheckerVerbose) {
      printf(" DRM readout timeout \n");
    }
#endif
  }

  /** loop over TRMs **/
  for (int itrm = 0; itrm < 10; ++itrm) {
    uint32_t slotId = itrm + 3;

    /** check current diagnostic word **/
    auto iword = mCheckerSummary.nDiagnosticWords;
    if (mCheckerSummary.DiagnosticWord[iword] & 0xFFFFFFF0) {
      mCheckerSummary.nDiagnosticWords++;
      iword++;
    }

    /** set current slot id **/
    mCheckerSummary.DiagnosticWord[iword] = slotId;

    /** check participating TRM **/
    if (!(partSlotMask & 1 << (itrm + 1))) {
      if (mDecoderSummary.trmDataHeader[itrm] != 0x0) {
        mCheckerSummary.DiagnosticWord[iword] |= DIAGNOSTIC_TRM_HEADER_UNEXPECTED;
#ifdef CHECKER_VERBOSE
        if (mCheckerVerbose) {
          printf(" Non-participating header found (slotId=%u) \n", slotId);
        }
#endif
      }
      continue;
    }

    /** check TRM Data Header **/
    if (!mDecoderSummary.trmDataHeader[itrm]) {
      mCheckerSummary.DiagnosticWord[iword] |= DIAGNOSTIC_TRM_HEADER_MISSING;
#ifdef CHECKER_VERBOSE
      if (mCheckerVerbose) {
        printf(" Missing TRM Data Header (slotId=%u) \n", slotId);
      }
#endif
      continue;
    }

    /** check TRM Data Trailer **/
    if (!mDecoderSummary.trmDataTrailer[itrm]) {
      mCheckerSummary.DiagnosticWord[iword] |= DIAGNOSTIC_TRM_TRAILER_MISSING;
#ifdef CHECKER_VERBOSE
      if (mCheckerVerbose) {
        printf(" Missing TRM Trailer (slotId=%u) \n", slotId);
      }
#endif
      mDecoderSummary.trmDataHeader[itrm] = 0x0;
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
    uint32_t eventCnt = GET_TRMDATAHEADER_EVENTCNT(mDecoderSummary.trmDataHeader[itrm]);
    if (eventCnt != locEvCnt % 1024) {
      mCheckerSummary.DiagnosticWord[iword] |= DIAGNOSTIC_TRM_EVENTCNT_MISMATCH;
#ifdef CHECKER_COUNTER
      mTRMCounters[itrm].EventCounterMismatch++;
#endif
#ifdef CHECKER_VERBOSE
      if (mCheckerVerbose) {
        printf(" TRM EventCounter / DRM LocalEventCounter mismatch: %u / %u (slotId=%u) \n", eventCnt, locEvCnt, slotId);
      }
#endif
    }

    /** check TRM empty bit **/
    if (GET_TRMDATAHEADER_EMPTYBIT(mDecoderSummary.trmDataHeader[itrm])) {
      mCheckerSummary.DiagnosticWord[iword] |= DIAGNOSTIC_TRM_EMPTYBIT_NOTZERO;
#ifdef CHECKER_COUNTER
      mTRMCounters[itrm].EBit++;
#endif
#ifdef CHECKER_VERBOSE
      if (mCheckerVerbose) {
        printf(" TRM empty bit is on (slotId=%u) \n", slotId);
      }
#endif
    }

    /** loop over TRM chains **/
    for (int ichain = 0; ichain < 2; ichain++) {

      /** check TRM Chain Header **/
      if (!mDecoderSummary.trmChainHeader[itrm][ichain]) {
        mCheckerSummary.DiagnosticWord[iword] |= DIAGNOSTIC_TRMCHAIN_HEADER_MISSING(ichain);
#ifdef CHECKER_VERBOSE
        if (mCheckerVerbose) {
          printf(" Missing TRM Chain Header (slotId=%u, chain=%d) \n", slotId, ichain);
        }
#endif
        continue;
      }

      /** check TRM Chain Trailer **/
      if (!mDecoderSummary.trmChainTrailer[itrm][ichain]) {
        mCheckerSummary.DiagnosticWord[iword] |= DIAGNOSTIC_TRMCHAIN_TRAILER_MISSING(ichain);
#ifdef CHECKER_VERBOSE
        if (mCheckerVerbose) {
          printf(" Missing TRM Chain Trailer (slotId=%u, chain=%d) \n", slotId, ichain);
        }
#endif
        mDecoderSummary.trmChainHeader[itrm][ichain] = 0x0;
        continue;
      }

      /** increment TRM Chain header counter **/
#ifdef CHECKER_COUNTER
      mTRMChainCounters[itrm][ichain].Headers++;
#endif

      /** check TDC errors **/
      if (mDecoderSummary.hasErrors[itrm][ichain]) {
        mCheckerSummary.DiagnosticWord[iword] |= DIAGNOSTIC_TRMCHAIN_TDCERROR_DETECTED(ichain);
#ifdef CHECKER_COUNTER
        mTRMChainCounters[itrm][ichain].TDCerror++;
#endif
#ifdef CHECKER_VERBOSE
        if (mCheckerVerbose) {
          printf(" TDC error detected (slotId=%u, chain=%d) \n", slotId, ichain);
        }
#endif
      }

      /** check TRM Chain event counter **/
      uint32_t eventCnt = GET_TRMCHAINTRAILER_EVENTCNT(mDecoderSummary.trmChainTrailer[itrm][ichain]);
      if (eventCnt != locEvCnt) {
        mCheckerSummary.DiagnosticWord[iword] |= DIAGNOSTIC_TRMCHAIN_EVENTCNT_MISMATCH(ichain);
#ifdef CHECKER_COUNTER
        mTRMChainCounters[itrm][ichain].EventCounterMismatch++;
#endif
#ifdef CHECKER_VERBOSE
        if (mCheckerVerbose) {
          printf(" TRM Chain EventCounter / DRM LocalEventCounter mismatch: %u / %u (slotId=%u, chain=%d) \n", eventCnt, locEvCnt, slotId, ichain);
        }
#endif
      }

      /** check TRM Chain Status **/
      uint32_t status = GET_TRMCHAINTRAILER_STATUS(mDecoderSummary.trmChainTrailer[itrm][ichain]);
      if (status != 0) {
        mCheckerSummary.DiagnosticWord[iword] |= DIAGNOSTIC_TRMCHAIN_STATUS_NOTZERO(ichain);
#ifdef CHECKER_COUNTER
        mTRMChainCounters[itrm][ichain].BadStatus++;
#endif
#ifdef CHECKER_VERBOSE
        if (mCheckerVerbose) {
          printf(" TRM Chain bad Status: %u (slotId=%u, chain=%d) \n", status, slotId, ichain);
        }
#endif
      }

      /** check TRM Chain BunchID **/
      uint32_t bunchCnt = GET_TRMCHAINHEADER_BUNCHCNT(mDecoderSummary.trmChainHeader[itrm][ichain]);
      if (bunchCnt != gbtBunchCnt) {
        mCheckerSummary.DiagnosticWord[iword] |= DIAGNOSTIC_TRMCHAIN_BUNCHCNT_MISMATCH(ichain);
#ifdef CHECKER_COUNTER
        mTRMChainCounters[itrm][ichain].BunchIDMismatch++;
#endif
#ifdef CHECKER_VERBOSE
        if (mCheckerVerbose) {
          printf(" TRM Chain BunchID / DRM L0BCID mismatch: %u / %u (slotId=%u, chain=%d) \n", bunchCnt, gbtBunchCnt, slotId, ichain);
        }
#endif
      }

      /** clear TRM chain summary data **/
      mDecoderSummary.trmChainHeader[itrm][ichain] = 0x0;
      mDecoderSummary.trmChainTrailer[itrm][ichain] = 0x0;

    } /** end of loop over TRM chains **/

    /** clear TRM summary data **/
    mDecoderSummary.trmDataHeader[itrm] = 0x0;
    mDecoderSummary.trmDataTrailer[itrm] = 0x0;

  } /** end of loop over TRMs **/

  /** check current diagnostic word **/
  auto iword = mCheckerSummary.nDiagnosticWords;
  if (mCheckerSummary.DiagnosticWord[iword] & 0xFFFFFFF0)
    mCheckerSummary.nDiagnosticWords++;

#ifdef CHECKER_VERBOSE
  if (mCheckerVerbose) {
    std::cout << colorBlue
              << "--- END CHECK EVENT: " << mCheckerSummary.nDiagnosticWords << " diagnostic words"
              << colorReset
              << std::endl;
  }
#endif

  /** clear DRM summary data **/
  mDecoderSummary.tofDataHeader = 0x0;
  mDecoderSummary.drmDataHeader = 0x0;
  mDecoderSummary.drmDataTrailer = 0x0;

  return false;
}

void Compressor::resetCounters()
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

void Compressor::checkSummary()
{
  char chname[2] = {'a', 'b'};

  std::cout << colorBlue
            << "--- SUMMARY COUNTERS: " << mEventCounter << " events "
            << " | " << mFatalCounter << " decode fatals "
            << " | " << mErrorCounter << " decode errors "
            << colorReset
            << std::endl;
  if (mEventCounter == 0)
    return;
  printf("\n");
  printf("    DRM ");
  float drmheaders = 100. * (float)mDRMCounters.Headers / (float)mEventCounter;
  printf("  \033%sheaders: %5.1f %%\033[0m ", drmheaders < 100. ? "[1;31m" : "[0m", drmheaders);
  if (mDRMCounters.Headers == 0)
    return;
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

} // namespace tof
} // namespace o2
