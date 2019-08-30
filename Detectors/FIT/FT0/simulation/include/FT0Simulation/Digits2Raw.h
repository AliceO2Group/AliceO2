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
#include <string>
#include <bitset>

namespace o2
{
namespace ft0
{
class Digits2Raw
{
  static constexpr int NCHANNELS_FT0 = 208;
  static constexpr int NCHANNELS_PM = 12;
  static constexpr int NPMs = 18;
  static constexpr float MV_2_NCHANNELS = 2.2857143;     //7 mV ->16channels
  static constexpr float CFD_NS_2_NCHANNELS = 76.804916; //1000.(ps)/13.02(channel);

 public:
  Digits2Raw() = default;
  //Digits2Raw(char * fileRaw, std::string fileDigitsName);
  void readDigits(const char* fileRaw, const char* fileDigitsName);
  void convertDigits(const o2::ft0::Digit& digit, const o2::ft0::LookUpTable& lut);
  static o2::ft0::LookUpTable linear()
  {
    std::vector<o2::ft0::Topo> lut_data(NCHANNELS_PM * NPMs);
    for (int link = 0; link < NPMs; ++link)
      for (int mcp = 0; mcp < NCHANNELS_PM; ++mcp)
        lut_data[link * NCHANNELS_PM + mcp] = o2::ft0::Topo{link, mcp};

    return o2::ft0::LookUpTable{lut_data};
  }

 private:
  void flushEvent(int link, o2::InteractionRecord const& mIntRecord, uint nchannels);
  void setGBTHeader(int link, o2::InteractionRecord const& mIntRecord, uint nchannels);
  void setRDH(int link, o2::InteractionRecord const& mIntRecord);
  std::ofstream mFileDest;
  //  FILE* mFileDest;
  o2::ft0::EventHeader mEventHeader;
  o2::ft0::EventData mEventData[NCHANNELS_FT0];
  o2::header::RAWDataHeader mRDH;
  /////////////////////////////////////////////////
  ClassDefNV(Digits2Raw, 1);
};

} // namespace ft0
} // namespace o2
#endif
