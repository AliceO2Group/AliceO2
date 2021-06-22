// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_TRDCOMPRESSEDDIGIT_H
#define ALICEO2_TRD_TRDCOMPRESSEDDIGIT_H

#include <cstdint>
#include <array>
#include "Rtypes.h" // for ClassDef
#include "DataFormatsTRD/Constants.h"
#include "fairlogger/Logger.h"
#include "gsl/span"

namespace o2
{
namespace trd
{

// Compressed Digit class for TRD
// Notes:
//      This is to simplify the handling of raw data that comes in in the same 3 timebins per 32 bit integer format.

class CompressedDigit
{
 public:
  CompressedDigit() = default;
  ~CompressedDigit() = default;
  CompressedDigit(const int det, const int rob, const int mcm, const int channel, const std::array<uint16_t, constants::TIMEBINS>& adc);
  CompressedDigit(const int det, const int rob, const int mcm, const int channel); // add adc data in a seperate step

  // Copy
  CompressedDigit(const CompressedDigit&) = default;
  // Assignment
  CompressedDigit& operator=(const CompressedDigit&) = default;
  // Modifiers
  void setChannel(int channel)
  {
    mHeader &= ~(0x003f);
    mHeader |= (channel)&0x003f;
  }
  void setMCM(int mcm)
  {
    mHeader &= ~(0x3c0);
    mHeader |= (mcm)&0x3c0;
  }
  void setROB(int rob)
  {
    mHeader &= ~(0x1c00);
    mHeader |= (rob)&0x1c00;
  }
  void setDetector(int det)
  {
    mHeader &= ~(0x7fe000);
    mHeader |= (det << 12) & 0x7fe000;
  }
  void setADC(std::array<uint16_t, constants::TIMEBINS> const& adcs)
  {
    int adcindex = 0;
    for (auto adc : adcs) {
      int rem = adcindex % 3;
      //    LOG(info) << "adc index :" << adcindex << " rem:" << rem << " adcindex/3=" << adcindex/3;
      mADC[adcindex / 3] &= ~((0x3ff) << (rem * 10));
      //     LOG(info) << "mADC[adcindex/3] after &= :" << std::hex << mADC[adcindex/3] << rem;
      mADC[adcindex / 3] |= (adcs[adcindex] << (rem * 10));
      //     LOG(info) << "mADC[adcindex/3] after  != :" << std::hex << mADC[adcindex/3] << rem;
      adcindex++;
    }
  }
  void setADCi(int index, uint16_t adcvalue)
  {
    int rem = index % 3;
    mADC[index / 3] &= ~((0x3ff) << (rem * 10));
    mADC[index / 3] |= (adcvalue << (rem * 10));
  }
  // Get methods
  int getChannel() const { return mHeader & 0x3f; }
  int getMCM() const { return (mHeader & 0x3c0) >> 6; }
  int getROB() const { return (mHeader & 0x1c00) >> 10; }
  int getDetector() const { return (mHeader & 0x7fe000) >> 12; }
  bool isSharedCompressedDigit() const;

  uint16_t operator[](const int index) { return mADC[index / 3] >> ((index % 3) * 10); }
  uint32_t getADCsum() const
  {
    uint32_t sum = 0;
    for (int i = 0; i < constants::TIMEBINS; ++i) {
      sum += (mADC[i / 3] >> ((i % 3) * 10));
    }
    return sum;
  }

 private:
  uint32_t mHeader;
  //             3322 2222 2222 1111 1111 1100 0000 0000
  //             1098 7654 3210 9876 5432 1098 7654 3210
  // uint32_t:   0000 0000 0DDD DDDD DDDR RRMM MMCC CCCC
  // C= channel[5 bits 0-21], M=MCM [4bits 0-15], R=ROB[3bits 0-7], D=Detector[10 bits 0-540]
  std::array<uint32_t, 10> mADC; // ADC vector (30 time-bins) packed into 10 32bit integers.
  ClassDefNV(CompressedDigit, 1);
};

} // namespace trd
} // namespace o2

#endif
