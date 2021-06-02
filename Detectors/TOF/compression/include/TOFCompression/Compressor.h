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

#ifndef O2_TOF_COMPRESSOR
#define O2_TOF_COMPRESSOR

#include <fstream>
#include <string>
#include <cstdint>
#include "Headers/RAWDataHeader.h"
#include "DataFormatsTOF/RawDataFormat.h"
#include "DataFormatsTOF/CompressedDataFormat.h"

namespace o2
{
namespace tof
{

template <typename RDH, bool verbose, bool paranoid>
class Compressor
{

 public:
  Compressor() { mDecoderSaveBuffer = new char[mDecoderSaveBufferSize]; };
  ~Compressor() { delete[] mDecoderSaveBuffer; };

  inline bool run()
  {
    rewind();
    if (mDecoderCONET) {
      mDecoderPointerMax = reinterpret_cast<const uint32_t*>(mDecoderBuffer + mDecoderBufferSize);
      while (mDecoderPointer < mDecoderPointerMax) {
        mEventCounter++;
        processDRM();
        if (mDecoderFatal) {
          mFatalCounter++;
        }
        if (mDecoderError) {
          mErrorCounter++;
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
    encoderRewind();
  };

  void checkSummary();
  void resetCounters();

  void setDecoderCONET(bool val)
  {
    mDecoderCONET = val;
    mDecoderNextWordStep = val ? 0 : 2;
  };

  void setDecoderVerbose(bool val) { mDecoderVerbose = val; };
  void setEncoderVerbose(bool val) { mEncoderVerbose = val; };
  void setCheckerVerbose(bool val) { mCheckerVerbose = val; };

  void setDecoderBuffer(const char* val) { mDecoderBuffer = val; };
  void setEncoderBuffer(char* val) { mEncoderBuffer = val; };
  void setDecoderBufferSize(long val) { mDecoderBufferSize = val; };
  void setEncoderBufferSize(long val) { mEncoderBufferSize = val; };

  inline uint32_t getDecoderByteCounter() const { return reinterpret_cast<const char*>(mDecoderPointer) - mDecoderBuffer; };
  inline uint32_t getEncoderByteCounter() const { return reinterpret_cast<char*>(mEncoderPointer) - mEncoderBuffer; };

  // benchmarks
  double mIntegratedBytes = 0.;
  double mIntegratedTime = 0.;

 protected:
  bool processHBF();
  bool processDRM();
  bool processLTM();
  bool processTRM();
  bool processTRMchain(int itrm, int ichain);

  /** decoder private functions and data members **/

  bool decoderParanoid();
  inline void decoderRewind() { mDecoderPointer = reinterpret_cast<const uint32_t*>(mDecoderBuffer); };
  inline void decoderNext()
  {
    mDecoderPointer += mDecoderNextWord;
    //    mDecoderNextWord = mDecoderNextWord == 1 ? 3 : 1;
    //    mDecoderNextWord = (mDecoderNextWord + 2) % 4;
    mDecoderNextWord = (mDecoderNextWord + mDecoderNextWordStep) & 0x3;
  };

  int mJumpRDH = 0;

  std::ifstream mDecoderFile;
  const char* mDecoderBuffer = nullptr;
  long mDecoderBufferSize;
  const uint32_t* mDecoderPointer = nullptr;
  const uint32_t* mDecoderPointerMax = nullptr;
  const uint32_t* mDecoderPointerNext = nullptr;
  uint8_t mDecoderNextWord = 1;
  uint8_t mDecoderNextWordStep = 2;
  const RDH* mDecoderRDH;
  bool mDecoderCONET = false;
  bool mDecoderVerbose = false;
  bool mDecoderError = false;
  bool mDecoderFatal = false;
  char* mDecoderSaveBuffer = nullptr;
  const int mDecoderSaveBufferSize = 33554432;
  uint32_t mDecoderSaveBufferDataSize = 0;
  uint32_t mDecoderSaveBufferDataLeft = 0;

  /** encoder private functions and data members **/

  void encoderSpider(int itrm);
  inline void encoderRewind() { mEncoderPointer = reinterpret_cast<uint32_t*>(mEncoderBuffer); };
  inline void encoderNext() { mEncoderPointer++; };

  std::ofstream mEncoderFile;
  char* mEncoderBuffer = nullptr;
  long mEncoderBufferSize;
  uint32_t* mEncoderPointer = nullptr;
  uint32_t* mEncoderPointerMax = nullptr;
  uint32_t* mEncoderPointerStart = nullptr;
  uint8_t mEncoderNextWord = 1;
  RDH* mEncoderRDH;
  bool mEncoderVerbose = false;

  /** checker private functions and data members **/

  bool checkerCheck();
  void checkerCheckRDH();

  uint32_t mEventCounter;
  uint32_t mFatalCounter;
  uint32_t mErrorCounter;
  bool mCheckerVerbose = false;

  struct DRMCounters_t {
    uint32_t Headers;
    uint32_t EventWordsMismatch;
    uint32_t clockStatus;
    uint32_t Fault;
    uint32_t RTOBit;
  } mDRMCounters = {0};

  struct TRMCounters_t {
    uint32_t Headers;
    uint32_t Empty;
    uint32_t EventCounterMismatch;
    uint32_t EventWordsMismatch;
    uint32_t EBit;
  } mTRMCounters[10] = {0};

  struct TRMChainCounters_t {
    uint32_t Headers;
    uint32_t EventCounterMismatch;
    uint32_t BadStatus;
    uint32_t BunchIDMismatch;
    uint32_t TDCerror;
  } mTRMChainCounters[10][2] = {0};

  /** summary data **/

  struct DecoderSummary_t {
    const uint32_t* tofDataHeader;
    const uint32_t* tofOrbit;
    const uint32_t* drmDataHeader;
    const uint32_t* drmHeadW1;
    const uint32_t* drmHeadW2;
    const uint32_t* drmHeadW3;
    const uint32_t* drmHeadW4;
    const uint32_t* drmHeadW5;
    const uint32_t* drmDataTrailer;
    const uint32_t* ltmDataHeader;
    const uint32_t* ltmDataTrailer;
    const uint32_t* trmDataHeader[10];
    const uint32_t* trmDataTrailer[10];
    const uint32_t* trmChainHeader[10][2];
    const uint32_t* trmChainTrailer[10][2];
    const uint32_t* trmDataHit[2][15][256];
    const uint32_t* trmError[10][2][32];
    uint8_t trmDataHits[2][15];
    uint8_t trmErrors[10][2];
    bool hasHits[10][2];
    bool hasErrors[10][2];
    bool drmDecodeError;
    bool ltmDecodeError;
    bool trmDecodeError[10];
  } mDecoderSummary = {nullptr};

  struct SpiderSummary_t {
    uint32_t FramePackedHit[256][256];
    uint8_t nFramePackedHits[256];
  } mSpiderSummary = {0};

  struct CheckerSummary_t {
    uint32_t nDiagnosticWords;
    uint32_t DiagnosticWord[12];
    uint32_t nTDCErrors;
  } mCheckerSummary = {0};
};

} // namespace tof
} // namespace o2

#endif /** O2_TOF_COMPRESSOR **/
