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
#include "Headers/RAWDataHeader.h"
#include "TBranch.h"
#include "TTree.h"

namespace o2
{
namespace ft0
{
class ReadRaw
{
  static constexpr int Nchannels_FT0 = 208;
  static constexpr int Nchannels_PM = 12;
  static constexpr int NPMs = 19;
  static constexpr int LinkTCM = 18;
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
  static o2::ft0::LookUpTable linear()
  {
    std::vector<o2::ft0::Topo> lut_data(Nchannels_PM * NPMs);
    for (int link = 0; link < NPMs; ++link)
      for (int mcp = 0; mcp < Nchannels_PM; ++mcp)
        lut_data[link * Nchannels_PM + mcp] = o2::ft0::Topo{link, mcp};

    return o2::ft0::LookUpTable{lut_data};
  }
  void printRDH(const o2::header::RAWDataHeader* h)
  {
    {
      if (!h) {
        printf("Provided RDH pointer is null\n");
        return;
      }
      printf("RDH| Ver:%2u Hsz:%2u Blgt:%4u FEEId:0x%04x PBit:%u\n",
             uint32_t(h->version), uint32_t(h->headerSize), uint32_t(h->blockLength), uint32_t(h->feeId), uint32_t(h->priority));
      printf("RDH|[CRU: Offs:%5u Msz:%4u LnkId:0x%02x Packet:%3u CRUId:0x%04x]\n",
             uint32_t(h->offsetToNext), uint32_t(h->memorySize), uint32_t(h->linkID), uint32_t(h->packetCounter), uint32_t(h->cruID));
      printf("RDH| TrgOrb:%9u HBOrb:%9u TrgBC:%4u HBBC:%4u TrgType:%u\n",
             uint32_t(h->triggerOrbit), uint32_t(h->heartbeatOrbit), uint32_t(h->triggerBC), uint32_t(h->heartbeatBC),

             uint32_t(h->triggerType));
      printf("RDH| DetField:0x%05x Par:0x%04x Stop:0x%04x PageCnt:%5u\n",
             uint32_t(h->detectorField), uint32_t(h->par), uint32_t(h->stop), uint32_t(h->pageCnt));
    }
  }

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

  ClassDefNV(ReadRaw, 1);
};

} // namespace ft0
} // namespace o2
#endif
