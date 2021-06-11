// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ReadRaw.h
/// \brief read raw data and writes digits
// Alla.Maevskaya@cern.ch

#ifndef ALICEO2_FT0_READRAW_H_
#define ALICEO2_FT0_READRAW_H_

#include <TStopwatch.h>
#include <array>
#include <bitset>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/DigitsTemp.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/LookUpTable.h"
#include "DataFormatsFT0/RawEventData.h"
#include "FT0Base/Geometry.h"
#include "Headers/RAWDataHeader.h"
#include "TBranch.h"
#include "TTree.h"

namespace o2
{
namespace ft0
{
class ReadRaw
{
  static constexpr int Nchannels_FT0 = o2::ft0::Geometry::Nchannels;
  static constexpr int Nchannels_PM = 12;
  static constexpr int NPMs = 20;
  //  static constexpr int LinkTCM = 19;
  static constexpr float MV_2_Nchannels = 2.2857143;     //7 mV ->16channels
  static constexpr float CFD_NS_2_Nchannels = 76.804916; //1000.(ps)/13.02(channel);
  //static constexpr int GBTWORDSIZE = 80;            //real size
  static constexpr int GBTWordSize = 128;            // with padding
  static constexpr int MaxGBTpacketBytes = 8 * 1024; // Max size of GBT packet in bytes (8KB)
  static constexpr int CRUWordSize = 16;

 public:
  ReadRaw() = default;
  ReadRaw(const std::string fileRaw, std::string fileDecodeData);
  void readData(const std::string fileRaw, const o2::ft0::LookUpTable& lut);
  void writeDigits(const std::string fileDecodeData);
  void close();

 private:
  std::ifstream mFileDest;
  o2::ft0::RawEventData mRawEventData;
  EventData mEventData[Nchannels_PM];
  bool mIsPadded = true;
  o2::ft0::EventHeader mEventHeader;
  o2::ft0::TCMdata mTCMdata;
  o2::ft0::Triggers mTrigger;
  char* mBuffer = nullptr;
  std::vector<char> mBufferLocal;
  long mSize;
  int mLinkTCM;
  std::map<o2::InteractionRecord, o2::ft0::DigitsTemp> mDigitAccum; // digit accumulator
  template <typename T>
  TBranch* getOrMakeBranch(TTree& tree, std::string brname, T* ptr)
  {
    if (auto br = tree.GetBranch(brname.c_str())) {
      br->SetAddress(static_cast<void*>(ptr));
      return br;
    }
    // otherwise make it
    return tree.Branch(brname.c_str(), ptr);
  }

  ClassDefNV(ReadRaw, 2);
};

} // namespace ft0
} // namespace o2
#endif
