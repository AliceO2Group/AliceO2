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
//file DataBlockFIT.h class  for RAW data format data blocks at FIT
//
// Artur.Furs
// afurs@cern.ch
//

#ifndef ALICEO2_FIT_DATABLOCKFIT_H_
#define ALICEO2_FIT_DATABLOCKFIT_H_
#include <iostream>
#include <Rtypes.h>
#include <FITRaw/DataBlockBase.h>

#include <gsl/span>
#include <iostream>
#include <cassert>
using namespace o2::fit;
namespace o2
{
namespace fit
{
//FIT DATA BLOCK DEFINITIONS

//standard data block from PM
template <typename RawHeaderPMtype, typename RawDataPMtype>
class DataBlockPM : public DataBlockBase<DataBlockPM, RawHeaderPMtype, RawDataPMtype>
{
 public:
  DataBlockPM() = default;
  DataBlockPM(const DataBlockPM&) = default;
  typedef RawHeaderPMtype RawHeaderPM;
  typedef RawDataPMtype RawDataPM;
  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    DataBlockWrapper<RawHeaderPM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderPM>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataPM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderPM>::mData[0].nGBTWords, srcByteShift);
  }
  std::vector<char> serialize() const
  {
    std::size_t nBytes = DataBlockWrapper<RawHeaderPM>::MaxNwords * SIZE_WORD;
    nBytes += DataBlockWrapper<RawHeaderPM>::mData[0].nGBTWords * SIZE_WORD;
    std::vector<char> vecBytes(nBytes);
    std::size_t destBytes = 0;
    DataBlockWrapper<RawHeaderPM>::serialize(vecBytes, DataBlockWrapper<RawHeaderPM>::MaxNwords, destBytes);
    DataBlockWrapper<RawDataPM>::serialize(vecBytes, DataBlockWrapper<RawHeaderPM>::mData[0].nGBTWords, destBytes);
    return vecBytes;
  }
  //Custom sanity checking for current deserialized block
  // put here code for raw data checking
  void sanityCheck(bool& flag)
  {
    if (DataBlockWrapper<RawDataPM>::mNelements == 0) {
      flag = false;
      return;
    }
    if (DataBlockWrapper<RawDataPM>::mNelements % 2 == 0 && DataBlockWrapper<RawDataPM>::mData[DataBlockWrapper<RawDataPM>::mNelements - 1].channelID == 0) {
      DataBlockWrapper<RawDataPM>::mNelements--; //in case of half GBT-word filling
    }
    //TODO, Descriptor checking, Channel range
  }
};

//standard data block from TCM
template <typename RawHeaderTCMtype, typename RawDataTCMtype>
class DataBlockTCM : public DataBlockBase<DataBlockTCM, RawHeaderTCMtype, RawDataTCMtype>
{
 public:
  DataBlockTCM() = default;
  DataBlockTCM(const DataBlockTCM&) = default;
  typedef RawHeaderTCMtype RawHeaderTCM;
  typedef RawDataTCMtype RawDataTCM;
  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    DataBlockWrapper<RawHeaderTCM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCM>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataTCM>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCM>::mData[0].nGBTWords, srcByteShift);
  }
  std::vector<char> serialize() const
  {
    std::size_t nBytes = DataBlockWrapper<RawHeaderTCM>::MaxNwords * SIZE_WORD;
    nBytes += DataBlockWrapper<RawHeaderTCM>::mData[0].nGBTWords * SIZE_WORD;
    std::vector<char> vecBytes(nBytes);
    std::size_t destBytes = 0;
    DataBlockWrapper<RawHeaderTCM>::serialize(vecBytes, DataBlockWrapper<RawHeaderTCM>::MaxNwords, destBytes);
    DataBlockWrapper<RawDataTCM>::serialize(vecBytes, DataBlockWrapper<RawHeaderTCM>::mData[0].nGBTWords, destBytes);
    return vecBytes;
  }
  //Custom sanity checking for current deserialized block
  // put here code for raw data checking
  void sanityCheck(bool& flag)
  {
    //TODO, Descriptor checking
  }
};

//extended TCM mode, 1 TCMdata + 8 TCMdataExtendedstructs
template <typename RawHeaderTCMextType, typename RawDataTCMtype, typename RawDataTCMextType>
class DataBlockTCMext : public DataBlockBase<DataBlockTCMext, RawHeaderTCMextType, RawDataTCMtype, RawDataTCMextType>
{
 public:
  DataBlockTCMext() = default;
  DataBlockTCMext(const DataBlockTCMext&) = default;
  typedef RawHeaderTCMextType RawHeaderTCMext;
  typedef RawDataTCMtype RawDataTCM;
  typedef RawDataTCMextType RawDataTCMext;

  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    DataBlockWrapper<RawHeaderTCMext>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCMext>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataTCM>::deserialize(srcBytes, DataBlockWrapper<RawDataTCM>::MaxNwords, srcByteShift);
    DataBlockWrapper<RawDataTCMext>::deserialize(srcBytes, DataBlockWrapper<RawHeaderTCMext>::mData[0].nGBTWords - DataBlockWrapper<RawDataTCM>::MaxNwords, srcByteShift);
  }

  std::vector<char> serialize() const
  {
    std::size_t nBytes = DataBlockWrapper<RawHeaderTCMext>::MaxNwords * SIZE_WORD;
    nBytes += DataBlockWrapper<RawHeaderTCMext>::mData[0].nGBTWords * SIZE_WORD;
    std::vector<char> vecBytes(nBytes);
    std::size_t destBytes = 0;
    DataBlockWrapper<RawHeaderTCMext>::serialize(vecBytes, DataBlockWrapper<RawHeaderTCMext>::MaxNwords, destBytes);
    DataBlockWrapper<RawDataTCM>::serialize(vecBytes, DataBlockWrapper<RawDataTCM>::MaxNwords, destBytes);
    DataBlockWrapper<RawDataTCMext>::serialize(vecBytes, DataBlockWrapper<RawHeaderTCMext>::mData[0].nGBTWords - DataBlockWrapper<RawDataTCM>::MaxNwords, destBytes);
    return vecBytes;
  }
  //Custom sanity checking for current deserialized block
  // put here code for raw data checking
  void sanityCheck(bool& flag)
  {

    //TODO, Descriptor checking
  }
};

} // namespace fit
} // namespace o2
#endif
