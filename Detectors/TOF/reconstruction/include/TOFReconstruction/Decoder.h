// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Decoder.h
/// \brief Definition of the TOF encoder

#ifndef ALICEO2_TOF_DECODER_H
#define ALICEO2_TOF_DECODER_H

#include <fstream>
#include <string>
#include <cstdint>
#include "DataFormatsTOF/CompressedDataFormat.h"
#include "TOFBase/Geo.h"
#include "TOFBase/Digit.h"
#include "TOFBase/Strip.h"
#include "TOFBase/WindowFiller.h"
#include <array>
#include "Headers/RAWDataHeader.h"

namespace o2
{
namespace tof
{
namespace compressed
{
/// \class Decoder
/// \brief Decoder class for TOF
///
class Decoder : public WindowFiller
{

 public:
  Decoder();
  ~Decoder() = default;

  bool open(std::string name);

  bool decode();
  void readTRM(int icru, int icrate, int orbit, int bunchid);

  bool close();
  void setVerbose(bool val) { mVerbose = val; };

  void printRDH() const;
  void printCrateInfo(int icru) const;
  void printTRMInfo(int icru) const;
  void printCrateTrailerInfo(int icru) const;
  void printHitInfo(int icru) const;

  static void fromRawHit2Digit(int icrate, int itrm, int itdc, int ichain, int channel, int orbit, int bunchid, int tdc, int tot, std::array<int, 6>& digitInfo); // convert raw info in digit info (channel, tdc, tot, bc), tdc = packetHit.time + (frameHeader.frameID << 13)

  char* nextPage(void* current, int shift = 8192);

 protected:
  static const int NCRU = 4;

  // benchmarks
  int mIntegratedBytes[NCRU];
  double mIntegratedTime = 0.;

  std::ifstream mFile[NCRU];
  bool mVerbose = false;
  bool mCruIn[NCRU];

  char* mBuffer[NCRU];
  std::vector<char> mBufferLocal;
  long mSize[NCRU];
  Union_t* mUnion[NCRU];
  Union_t* mUnionEnd[NCRU];

  int mHitDecoded = 0;

  o2::header::RAWDataHeader* mRDH;
};

} // namespace compressed
} // namespace tof
} // namespace o2
#endif
