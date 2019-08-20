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
#include "DataFormatsTOF/DataFormat.h"
#include "TOFBase/Geo.h"
#include "TOFBase/Digit.h"
#include <array>

namespace o2
{
namespace tof
{
namespace compressed
{
/// \class Decoder
/// \brief Decoder class for TOF
///
class Decoder
{

 public:
  Decoder() = default;
  ~Decoder() = default;

  bool open(std::string name);

  bool decode(std::vector<Digit>* digits);
  void readTRM(std::vector<Digit>* digits, int iddl, int orbit, int bunchid);

  bool close();
  void setVerbose(bool val) { mVerbose = val; };

  void printCrateInfo() const;
  void printTRMInfo() const;
  void printCrateTrailerInfo() const;
  void printHitInfo() const;

  static void fromRawHit2Digit(int iddl, int itrm, int itdc, int ichain, int channel, int orbit, int bunchid, int tdc, int tot, std::array<int, 4>& digitInfo); // convert raw info in digit info (channel, tdc, tot, bc), tdc = packetHit.time + (frameHeader.frameID << 13)

 protected:
  // benchmarks
  double mIntegratedBytes = 0.;
  double mIntegratedTime = 0.;

  std::ifstream mFile;
  bool mVerbose = false;

  char* mBuffer = nullptr;
  std::vector<char> mBufferLocal;
  long mSize;
  Union_t* mUnion;
  Union_t* mUnionEnd;
};

} // namespace compressed
} // namespace tof
} // namespace o2
#endif
