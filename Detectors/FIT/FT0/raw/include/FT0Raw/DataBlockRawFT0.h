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
//file DataBlockRawFT0.h class  for RAW data format data blocks at FT0
//
// Artur.Furs
// afurs@cern.ch
//  need to use references on the DataBlock fileds, with fast access
//  traites for classes and structs
//

#ifndef ALICEO2_FT0_DATABLOCKRAW_H_
#define ALICEO2_FT0_DATABLOCKRAW_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "CommonDataFormat/InteractionRecord.h"
#include <DataFormatsFT0/RawEventData.h>
#include <FITRaw/DataBlockRaw.h>

#include <gsl/span>
#include <boost/mpl/inherit.hpp>
#include <boost/mpl/vector.hpp>

#include <vector>
#include <tuple>
#include <array>
#include <iostream>
#include <cassert>
using namespace o2::fit;
namespace o2
{
namespace ft0
{
using RawHeaderPM = o2::ft0::EventHeader;
using RawDataPM = o2::ft0::EventData;
using RawHeaderTCM = o2::ft0::EventHeader;
using RawDataTCM = o2::ft0::TCMdata;
using RawHeaderTCMext = o2::ft0::EventHeader;
using RawDataTCMext = o2::ft0::TCMdataExtended;

using namespace std;

//FT0 DATA BLOCK DEFINITIONS

//standard data block from PM
class DataBlockPM : public o2::fit::DataBlockBase<DataBlockPM, RawHeaderPM, RawDataPM>
{
 public:
  DataBlockPM() = default;
  DataBlockPM(const DataBlockPM&) = default;
  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    DataBlockWrapper<RawHeaderPM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderPM>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataPM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderPM>::mData[0].nGBTWords, srcByteShift);
  }
  //Custom sanity checking for current deserialized block
  // put here code for raw data checking
  void sanityCheck(bool& flag)
  {
    if (DataBlockWrapper<RawDataPM>::mNelements % 2 == 0 && DataBlockWrapper<RawDataPM>::mData[DataBlockWrapper<RawDataPM>::mNelements - 1].channelID == 0)
      DataBlockWrapper<RawDataPM>::mNelements--; //in case of half GBT-word filling
    //TODO, Descriptor checking, Channel range
  }
};

//standard data block from TCM
class DataBlockTCM : public o2::fit::DataBlockBase<DataBlockTCM, RawHeaderTCM, RawDataTCM>
{
 public:
  DataBlockTCM() = default;
  DataBlockTCM(const DataBlockTCM&) = default;
  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    DataBlockWrapper<RawHeaderTCM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCM>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataTCM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCM>::mData[0].nGBTWords, srcByteShift);
  }
  //Custom sanity checking for current deserialized block
  // put here code for raw data checking
  void sanityCheck(bool& flag)
  {
    //TODO, Descriptor checking, Channel range
  }
};

//extended TCM mode, 1 TCMdata + 8 TCMdataExtendedstructs
class DataBlockTCMext : public o2::fit::DataBlockBase<DataBlockTCMext, RawHeaderTCMext, RawDataTCM, RawDataTCMext>
{
 public:
  DataBlockTCMext() = default;
  DataBlockTCMext(const DataBlockTCMext&) = default;
  //virtual ~DataBlockTCMext() = default;
  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    DataBlockWrapper<RawHeaderTCMext>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCMext>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataTCM>::deserialize(srcBytes, DataBlockWrapper<RawDataTCM>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataTCMext>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCMext>::mData[0].nGBTWords - DataBlockWrapper<RawDataTCM>::MaxNwords, srcByteShift);
  }

  //Custom sanity checking for current deserialized block
  // put here code for raw data checking
  void sanityCheck(bool& flag)
  {
    //TODO, Descriptor checking, Channel range
  }
};

} // namespace ft0
} // namespace o2
#endif