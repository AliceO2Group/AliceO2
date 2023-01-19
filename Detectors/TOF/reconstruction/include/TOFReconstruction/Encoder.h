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

/// \file Encoder.h
/// \brief Definition of the TOF encoder

#ifndef ALICEO2_TOF_ENCODER_H
#define ALICEO2_TOF_ENCODER_H

#include <fstream>
#include <string>
#include <cstdint>
#include "DataFormatsTOF/RawDataFormat.h"
#include "TOFBase/Geo.h"
#include "TOFBase/Digit.h"
#include "Headers/RAWDataHeader.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"

namespace o2
{
namespace tof
{
namespace raw
{

/// \class Encoder
/// \brief Encoder class for TOF
///
class Encoder
{

 public:
  Encoder();
  ~Encoder() = default;

  bool open(const std::string& name, const std::string& path = ".", const std::string& fileFor = "cruendpoint");
  bool alloc(long size);

  bool encode(std::vector<std::vector<o2::tof::Digit>> digitWindow, int tofwindow = 0);
  void encodeTRM(const std::vector<Digit>& summary, Int_t icrate, Int_t itrm, int& istart); // return next trm index

  bool flush(int icrate);
  bool close();
  void setVerbose(bool val) { mVerbose = val; };

  void setEncoderCRUZEROES(bool val = true) { mOldFormat = val; }

  int getSize(void* first, void* last);

  void nextWord(int icrate);

  void setContinuous(bool value) { mIsContinuous = value; }
  bool isContinuous() const { return mIsContinuous; }

  auto& getWriter() { return mFileWriter; };

  static int getNCRU() { return NCRU; }

 protected:
  // benchmarks
  double mIntegratedAllBytes = 0;
  double mIntegratedTime = 0.;

  static constexpr int NCRU = 4;
  static constexpr int NLINKSPERCRU = 72 / NCRU;

  bool mVerbose = false;

  char* mBuffer[72];
  std::vector<char> mBufferLocal;
  long mSize;
  Union_t* mUnion[72];
  Union_t* mStart[72];
  TOFDataHeader_t* mTOFDataHeader[72];
  DRMDataHeader_t* mDRMDataHeader[72];
  bool mNextWordStatus[72];

  bool mIsContinuous = true;

  o2::header::RAWDataHeader* mRDH[72];
  const o2::raw::HBFUtils& mHBFSampler = o2::raw::HBFUtils::Instance();
  o2::raw::RawFileWriter mFileWriter{o2::header::gDataOriginTOF};

  bool mCrateOn[72];

  bool mStartRun = true;
  int mFirstBC = 0;

  bool mOldFormat = false;

  // temporary variable for encoding
  int mEventCounter;         //!
  o2::InteractionRecord mIR; //!
};

} // namespace compressed
} // namespace tof
} // namespace o2
#endif
