// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digits2Raw.h
/// \brief converts digits to raw format
// Alla.Maevskaya@cern.ch

#ifndef ALICEO2_FT0_DIGITS2RAW_H_
#define ALICEO2_FT0_DIGITS2RAW_H_

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFT0/RawEventData.h"
#include "DataFormatsFT0/LookUpTable.h"
#include "DataFormatsFT0/Digit.h"
#include <TStopwatch.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <array>
#include <string>
#include <bitset>
#include <vector>

namespace o2
{
namespace ft0
{
class Digits2Raw
{

  static constexpr int Nchannels_FT0 = 208;
  static constexpr int Nchannels_PM = 12;
  static constexpr int NPMs = 18;
  //  static constexpr int GBTWORDSIZE = 80;            //real size
  static constexpr int GBTWordSize = 128; // with padding
  static constexpr int Max_Page_size = 8192;

 public:
  Digits2Raw() = default;
  Digits2Raw(const std::string fileRaw, std::string fileDigitsName);
  void readDigits(const std::string fileDigitsName);
  void convertDigits(const o2::ft0::Digit& digit, const o2::ft0::LookUpTable& lut);
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
  std::ofstream mFileDest;
  std::array<o2::ft0::DataPageWriter, NPMs> mPages;
  o2::ft0::RawEventData mRawEventData;

  /////////////////////////////////////////////////

  ClassDefNV(Digits2Raw, 1);
};

} // namespace ft0
} // namespace o2
#endif
