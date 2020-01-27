// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  bool open(std::string name);
  bool alloc(long size);

  bool encode(std::vector<std::vector<o2::tof::Digit>> digitWindow, int tofwindow = 0);
  void encodeTRM(const std::vector<Digit>& summary, Int_t icrate, Int_t itrm, int& istart); // return next trm index

  void openRDH(int icrate);
  void addPage(int icrate);
  void closeRDH(int icrate);

  bool flush();
  bool flush(int icrate);
  bool close();
  void setVerbose(bool val) { mVerbose = val; };

  char* nextPage(void* current, int step);
  int getSize(void* first, void* last);

  void nextWord(int icrate);
  void nextWordNoEmpty(int icrate);

  void setContinuous(bool value) { mIsContinuous = value; }
  bool isContinuous() const { return mIsContinuous; }

 protected:
  // benchmarks
  double mIntegratedBytes[72];
  double mIntegratedAllBytes = 0;
  double mIntegratedTime = 0.;

  static constexpr int NCRU = 4;
  static constexpr int NLINKSPERCRU = 72 / NCRU;
  std::ofstream mFileCRU[NCRU];

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
  o2::raw::HBFUtils mHBFSampler;
  int mNRDH[72];

  bool mStartRun = true;

  // temporary variable for encoding
  int mEventCounter;         //!
  o2::InteractionRecord mIR; //!
};

} // namespace compressed
} // namespace tof
} // namespace o2
#endif
