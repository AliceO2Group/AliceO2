// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//file RawEventData.h class  for RAW data format
//Alla.Maevskaya@cern.ch
// with Artur.Furs
//
#ifndef ALICEO2_FT0_RAWEVENTDATA_H_
#define ALICEO2_FT0_RAWEVENTDATA_H_

#include "Headers/RAWDataHeader.h"
#include "TList.h" //temporary for QC-FT0 (ChannelTimeCalibrationCheck.cxx), should be moved
#include "DataFormatsFIT/RawEventData.h"
#include "FT0Base/Geometry.h"
#include <CommonDataFormat/InteractionRecord.h>
#include <Framework/Logger.h>
#include <utility>
#include <cstring>
#include <iomanip>
#include "Rtypes.h"

namespace o2
{
namespace ft0
{
constexpr int Nchannels_FT0 = o2::ft0::Geometry::Nchannels;
constexpr int Nchannels_PM = 12;
constexpr int NPMs = 20;
using EventHeader = o2::fit::EventHeader;
using EventData = o2::fit::EventData;
using TCMdata = o2::fit::TCMdata;
using TCMdataExtended = o2::fit::TCMdataExtended;
class RawEventData
{
 public:
  RawEventData() = default;
  void print() const;
  const static int gStartDescriptor = 0x0000000f;
  static const size_t sPayloadSizeSecondWord = 11;
  static const size_t sPayloadSizeFirstWord = 5;
  static constexpr size_t sPayloadSize = 16;
  int size() const
  {
    return 1                         // EventHeader
           + mEventHeader.nGBTWords; // EventData
  }

  std::vector<char> to_vector(bool tcm)
  {
    constexpr int CRUWordSize = 16;
    const char padding[CRUWordSize] = {0};

    std::vector<char> result(size() * CRUWordSize);
    char* out = result.data();
    if (!tcm) {
      std::memcpy(out, &mEventHeader, sPayloadSize);
      out += sPayloadSize;
      LOG(DEBUG) << "write header words " << (int)mEventHeader.nGBTWords << " orbit " << int(mEventHeader.orbit) << " bc " << int(mEventHeader.bc) << " out " << result.size();
      if (mIsPadded) {
        out += CRUWordSize - sPayloadSize;
      }
      for (int i = 0; i < mEventHeader.nGBTWords; ++i) {
        std::memcpy(out, &mEventData[2 * i], sPayloadSizeFirstWord);
        LOG(DEBUG) << " 1st word " << mEventData[2 * i].channelID << " charge " << mEventData[2 * i].charge << " time " << mEventData[2 * i].time << " out " << result.size();
        out += sPayloadSizeFirstWord;
        std::memcpy(out, &mEventData[2 * i + 1], sPayloadSizeSecondWord);
        out += sPayloadSizeSecondWord;
        LOG(DEBUG) << " 2nd word " << mEventData[2 * i + 1].channelID << " charge " << mEventData[2 * i + 1].charge << " time " << mEventData[2 * i + 1].time << " out " << result.size();
        if (mIsPadded) {
          out += CRUWordSize - sPayloadSizeSecondWord - sPayloadSizeFirstWord;
        }
      }
    } else {
      // TCM data
      std::memcpy(out, &mEventHeader, sPayloadSize);
      out += sPayloadSize;
      LOG(DEBUG) << "write TCM header words " << (int)mEventHeader.nGBTWords << " orbit " << int(mEventHeader.orbit) << " bc " << int(mEventHeader.bc) << " out " << result.size();
      std::memcpy(out, &mTCMdata, sizeof(TCMdata));
      out += sizeof(TCMdata);
      LOG(DEBUG) << "write TCM words " << sizeof(mTCMdata) << " orbit " << int(mEventHeader.orbit) << " bc " << int(mEventHeader.bc) << " out " << result.size() << " sum time A " << mTCMdata.timeA;
    }

    return result;
  }
  void setIsPadded(bool isPadding128)
  {
    mIsPadded = isPadding128;
  }

 public:
  EventHeader mEventHeader;           //!
  EventData mEventData[Nchannels_PM]; //!
  TCMdata mTCMdata;                   //!
  bool mIsPadded = true;
  /////////////////////////////////////////////////
  ClassDefNV(RawEventData, 2);
};
std::ostream& operator<<(std::ostream& stream, const RawEventData& data);

class DataPageWriter
{
  std::vector<char> mBuffer;
  int mNpacketsInBuffer = 0;
  std::vector<std::vector<char>> mPages;
  std::vector<int> mNpackets;
  static constexpr int MAX_Page_size = 8192;

 public:
  o2::header::RAWDataHeader mRDH;
  void flush(std::ostream& str)
  {
    writePage();
    mRDH.stop = 0;
    for (int page = 0; page < int(mPages.size()); ++page) {
      mRDH.memorySize = mPages[page].size() + mRDH.headerSize;
      mRDH.offsetToNext = mRDH.memorySize;
      mRDH.packetCounter = mNpackets[page];
      str.write(reinterpret_cast<const char*>(&mRDH), sizeof(mRDH));
      str.write(mPages[page].data(), mPages[page].size());
      mRDH.pageCnt++;
      LOG(INFO) << " header " << mRDH.linkID << " end " << mRDH.endPointID;
    }
    if (!mPages.empty()) {
      mRDH.memorySize = mRDH.headerSize;
      mRDH.offsetToNext = mRDH.memorySize;
      mRDH.stop = 1;
      mRDH.pageCnt++;
      str.write(reinterpret_cast<const char*>(&mRDH), sizeof(mRDH));
      mPages.clear();
      mNpackets.clear();
    }
  }

  void writePage()
  {
    if (mBuffer.size() == 0) {
      return;
    }
    mPages.emplace_back(std::move(mBuffer));
    LOG(DEBUG) << " writePage " << mBuffer.size();
    mNpackets.push_back(mNpacketsInBuffer);
    mNpacketsInBuffer = 0;
    mBuffer.clear();
  }

  void write(std::vector<char> const& new_data)
  {
    if (mBuffer.size() + new_data.size() + mRDH.headerSize > MAX_Page_size) {
      LOG(DEBUG) << " write rest " << mBuffer.size() << " " << new_data.size() << " " << mRDH.headerSize;
      writePage();
    }
    LOG(DEBUG) << "  write vector " << new_data.size() << " buffer " << mBuffer.size() << " RDH " << mRDH.headerSize << " new data " << new_data.data();
    mBuffer.insert(mBuffer.end(), new_data.begin(), new_data.end());
    mNpacketsInBuffer++;
    LOG(DEBUG) << "  write vector end mBuffer.size " << mBuffer.size() << " mNpacketsInBuffer " << mNpacketsInBuffer << " newdtata " << new_data.size();
  }
};
} // namespace ft0
} // namespace o2
#endif
