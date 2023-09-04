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

/// \file RawDataDecoder.h
/// \brief Digits  tw Raw translation
/// \author Roman Lietava

#ifndef ALICEO2_CTP_RAWDATADECODER_H_
#define ALICEO2_CTP_RAWDATADECODER_H_

#include <vector>
#include <map>
#include <deque>
#include "Framework/InputRecord.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsCTP/LumiInfo.h"

namespace o2
{
namespace ctp
{
class RawDataDecoder
{
 public:
  RawDataDecoder() = default;
  ~RawDataDecoder() = default;
  static void makeGBTWordInverse(std::vector<gbtword80_t>& diglets, gbtword80_t& GBTWord, gbtword80_t& remnant, uint32_t& size_gbt, uint32_t Npld);
  int addCTPDigit(uint32_t linkCRU, uint32_t triggerOrbit, gbtword80_t& diglet, gbtword80_t& pldmask, std::map<o2::InteractionRecord, CTPDigit>& digits);
  int decodeRaw(o2::framework::InputRecord& inputs, std::vector<o2::framework::InputSpec>& filter, o2::pmr::vector<CTPDigit>& digits, std::vector<LumiInfo>& lumiPointsHBF1);
  int decodeRaw(o2::framework::InputRecord& inputs, std::vector<o2::framework::InputSpec>& filter, std::vector<CTPDigit>& digits, std::vector<LumiInfo>& lumiPointsHBF1);
  void setDecodeInps(bool decodeinps) { mDecodeInps = decodeinps; }
  void setDoLumi(bool lumi) { mDoLumi = lumi; }
  void setDoDigits(bool digi) { mDoDigits = digi; }
  void setVerbose(bool v) { mVerbose = v; }
  void setMAXErrors(int m) { mErrorMax = m; }
  int setLumiInp(int lumiinp, std::string inp);
  uint32_t getIRRejected() const { return mIRRejected; }
  uint32_t getTCRRejected() const { return mTCRRejected; }
  std::vector<uint32_t>& getTFOrbits() { return mTFOrbits; }
  int getErrorIR() { return mErrorIR; }
  int getErrorTCR() { return mErrorTCR; }
  int init();
  static int shiftNew(const o2::InteractionRecord& irin, uint32_t TFOrbit, std::bitset<48>& inpmask, int64_t shift, int level, std::map<o2::InteractionRecord, CTPDigit>& digmap);
  static int shiftInputs(std::map<o2::InteractionRecord, CTPDigit>& digitsMap, o2::pmr::vector<CTPDigit>& digits, uint32_t TFOrbit);

 private:
  static constexpr uint32_t TF_TRIGGERTYPE_MASK = 0x800;
  static constexpr uint32_t HB_TRIGGERTYPE_MASK = 0x2;
  // true: full inps decoding includine latency shifts here; false: latency shifts in CTF decoder
  bool mDecodeInps = false;
  // for digits
  bool mDoDigits = true;
  std::vector<CTPDigit> mOutputDigits;
  // for lumi
  bool mDoLumi = true;
  //
  static constexpr std::bitset<o2::ctp::CTP_NINPUTS> LMMASKInputs = 0xfff;
  static constexpr std::bitset<o2::ctp::CTP_NINPUTS> L0MASKInputs = 0xfff000;
  static constexpr std::bitset<o2::ctp::CTP_NINPUTS> L1MASKInputs = (0xffffffull << 24);
  gbtword80_t mTVXMask = 0x4;  // TVX is 3rd input
  gbtword80_t mVBAMask = 0x20; // VBA is 6 th input
  bool mVerbose = false;
  uint32_t mIRRejected = 0;
  uint32_t mTCRRejected = 0;
  bool mPadding = true;
  uint32_t mTFOrbit = 0;
  std::vector<uint32_t> mTFOrbits;
  // error verbosness
  int mErrorIR = 0;
  int mErrorTCR = 0;
  int mErrorMax = 3;
  bool mStickyError = false;
};
} // namespace ctp
} // namespace o2
#endif
