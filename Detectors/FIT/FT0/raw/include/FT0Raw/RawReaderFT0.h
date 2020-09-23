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
//file RawReaderFT0.h class  for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode FT0 data blocks and push them to DigitBlockFT0 for proccess

#ifndef ALICEO2_FIT_RAWREADERFT0_H_
#define ALICEO2_FIT_RAWREADERFT0_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "FT0Raw/DataBlockFT0.h"
#include "FT0Raw/DigitBlockFT0.h"
#include "FITRaw/RawReaderBase.h"

#include <boost/mpl/inherit.hpp>
#include <boost/mpl/vector.hpp>

#include <CommonDataFormat/InteractionRecord.h>
#include "Headers/RAWDataHeader.h"
//#include "DataFormatsFT0/Digit.h"
//#include "DataFormatsFT0/ChannelData.h"

#include <gsl/span>
namespace o2
{
namespace ft0
{


// Common raw reader for FT0
template <class DigitBlockFT0type,class DataBlockPMtype,class DataBlockTCMtype>
class RawReaderFT0Base : public RawReaderBase<DigitBlockFT0type>
{
 public:
  typedef RawReaderBase<DigitBlockFT0type> RawReaderBaseType;
  RawReaderFT0Base() = default;
  ~RawReaderFT0Base() = default;
  //deserialize payload to raw data blocks and proccesss them to digits
  void process(int linkID, gsl::span<const uint8_t> payload)
  {
    if (0 <= linkID && linkID < 18) {
      //PM data proccessing
      RawReaderBaseType::template processBinaryData<DataBlockPMtype>(payload, linkID);
    } else if (linkID == 18) {
      //TCM data proccessing
      RawReaderBaseType::template processBinaryData<DataBlockTCMtype>(payload, linkID);
    } else {
      //put here code in case of bad rdh.linkID value
      LOG(INFO) << "WARNING! WRONG LINK ID!";
      return;
    }

    //
  }
};
/*
template <bool IsExtendedMode>
class RawReaderFT0:public typename std::enable_if<IsExtendedMode,RawReaderFT0Base<DigitBlockFT0,DataBlockPM,DataBlockTCM>>::type {};


template <bool IsExtendedMode>
class RawReaderFT0:public typename std::enable_if<!IsExtendedMode,RawReaderFT0Base<DigitBlockFT0ext,DataBlockPM,DataBlockTCMext>>::type {};
*/
using RawReaderFT0normMode = RawReaderFT0Base<DigitBlockFT0,DataBlockPM,DataBlockTCM>;

using RawReaderFT0extMode = RawReaderFT0Base<DigitBlockFT0ext,DataBlockPM,DataBlockTCMext>;

template<bool IsExtendedMode>
using RawReaderFT0 = typename std::conditional<IsExtendedMode,RawReaderFT0extMode,RawReaderFT0normMode>::type;

} // namespace ft0
} // namespace o2

#endif