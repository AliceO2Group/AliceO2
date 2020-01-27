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

class Compressor
{

 public:
  Compressor() = default;
  ~Compressor();

  inline bool run()
  {
    rewind();
    while (!processHBF())
      ;
    return false;
  };

  bool init();
  bool open(const std::string inFileName, const std::string outFileName);
  bool close();
  inline bool read() { return decoderRead(); };
  inline void rewind()
  {
    decoderRewind();
    encoderRewind();
  };
  inline bool write() { return encoderWrite(); };

  void checkSummary();
  void resetCounters();

  void setDecoderVerbose(bool val) { mDecoderVerbose = val; };
  void setEncoderVerbose(bool val) { mEncoderVerbose = val; };
  void setCheckerVerbose(bool val) { mCheckerVerbose = val; };

  void setDecoderBuffer(char* val) { mDecoderBuffer = val; };
  void setEncoderBuffer(char* val) { mEncoderBuffer = val; };
  void setDecoderBufferSize(long val) { mDecoderBufferSize = val; };
  void setEncoderBufferSize(long val) { mEncoderBufferSize = val; };

  inline uint32_t getDecoderByteCounter() const { return reinterpret_cast<char*>(mDecoderPointer) - mDecoderBuffer; };
  inline uint32_t getEncoderByteCounter() const { return reinterpret_cast<char*>(mEncoderPointer) - mEncoderBuffer; };

  // benchmarks
  double mIntegratedBytes = 0.;
  double mIntegratedTime = 0.;

 protected:
  bool processHBF();
  bool processDRM();

  /** decoder private functions and data members **/

  bool decoderInit();
  bool decoderOpen(const std::string name);
  bool decoderRead();
  bool decoderClose();
  bool decoderParanoid();
  inline void decoderRewind() { mDecoderPointer = reinterpret_cast<uint32_t*>(mDecoderBuffer); };
  inline void decoderNext()
  {
    mDecoderPointer += mDecoderNextWord;
    //    mDecoderNextWord = mDecoderNextWord == 1 ? 3 : 1;
    //    mDecoderNextWord = (mDecoderNextWord + 2) % 4;
    mDecoderNextWord = (mDecoderNextWord + 2) & 0x3;
  };

  int mJumpRDH = 0;

  std::ifstream mDecoderFile;
  char* mDecoderBuffer = nullptr;
  bool mOwnDecoderBuffer = false;
  long mDecoderBufferSize = 8192;
  //  long mDecoderBufferSize = 1048576;
  uint32_t* mDecoderPointer = nullptr;
  uint32_t* mDecoderPointerMax = nullptr;
  uint32_t* mDecoderPointerNext = nullptr;
  uint8_t mDecoderNextWord = 1;
  o2::header::RAWDataHeader* mDecoderRDH;
  bool mDecoderVerbose = false;
  bool mDecoderError = false;
  bool mDecoderFatal = false;
  char mDecoderSaveBuffer[1048576];
  uint32_t mDecoderSaveBufferDataSize = 0;
  uint32_t mDecoderSaveBufferDataLeft = 0;

  /** encoder private functions and data members **/

  bool encoderInit();
  bool encoderOpen(const std::string name);
  bool encoderWrite();
  bool encoderClose();
  void encoderSpider(int itrm);
  inline void encoderRewind() { mEncoderPointer = reinterpret_cast<uint32_t*>(mEncoderBuffer); };
  inline void encoderNext() { mEncoderPointer++; };

  std::ofstream mEncoderFile;
  char* mEncoderBuffer = nullptr;
  bool mOwnEncoderBuffer = false;
  long mEncoderBufferSize = 1048576;
  uint32_t* mEncoderPointer = nullptr;
  uint32_t* mEncoderPointerMax = nullptr;
  uint32_t* mEncoderPointerStart = nullptr;
  uint8_t mEncoderNextWord = 1;
  o2::header::RAWDataHeader* mEncoderRDH;
  bool mEncoderVerbose = false;

  /** checker private functions and data members **/

  bool checkerCheck();

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
    uint32_t tofDataHeader;
    uint32_t tofOrbit;
    uint32_t drmDataHeader;
    uint32_t drmHeadW1;
    uint32_t drmHeadW2;
    uint32_t drmHeadW3;
    uint32_t drmHeadW4;
    uint32_t drmHeadW5;
    uint32_t drmDataTrailer;
    uint32_t trmDataHeader[10];
    uint32_t trmDataTrailer[10];
    uint32_t trmChainHeader[10][2];
    uint32_t trmChainTrailer[10][2];
    uint32_t trmDataHit[2][15][256];
    uint8_t trmDataHits[2][15];
    bool hasHits[10][2];
    bool hasErrors[10][2];
    bool decodeError;
  } mDecoderSummary = {0};

  struct SpiderSummary_t {
    uint32_t FramePackedHit[256][256];
    uint8_t nFramePackedHits[256];
  } mSpiderSummary = {0};

  struct CheckerSummary_t {
    uint32_t nDiagnosticWords;
    uint32_t DiagnosticWord[12];
  } mCheckerSummary = {0};
};

} // namespace tof
} // namespace o2

#endif /** O2_TOF_COMPRESSOR **/
