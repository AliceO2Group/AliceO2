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
//
// file DataBlockFIT.h class  for RAW data format data blocks at FIT
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
// FIT DATA BLOCK DEFINITIONS

// standard data block from PM
template <typename ConfigType, typename RawHeaderPMtype, typename RawDataPMtype>
class DataBlockPM : public DataBlockBase<DataBlockPM, ConfigType, RawHeaderPMtype, RawDataPMtype>
{
 public:
  DataBlockPM() = default;
  DataBlockPM(const DataBlockPM&) = default;
  typedef RawHeaderPMtype RawHeaderPM;
  typedef RawDataPMtype RawDataPM;
  typedef DataBlockWrapper<ConfigType, RawHeaderPM> HeaderPM;
  typedef DataBlockWrapper<ConfigType, RawDataPM> DataPM;

  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    HeaderPM::deserialize(srcBytes, HeaderPM::MaxNwords, srcByteShift);
    DataPM::deserialize(srcBytes, HeaderPM::mData[0].nGBTWords, srcByteShift);
  }
  const std::size_t getNgbtWords() const { return HeaderPM::mData[0].nGBTWords; }
  std::vector<char> serialize() const
  {
    std::size_t nBytes = HeaderPM::MaxNwords * HeaderPM::sSizeWord;
    nBytes += HeaderPM::mData[0].nGBTWords * HeaderPM::sSizeWord;
    std::vector<char> vecBytes(nBytes);
    std::size_t destBytes = 0;
    HeaderPM::serialize(vecBytes, HeaderPM::MaxNwords, destBytes);
    DataPM::serialize(vecBytes, HeaderPM::mData[0].nGBTWords, destBytes);
    return vecBytes;
  }
  // Custom sanity checking for current deserialized block
  //  put here code for raw data checking
  void sanityCheck(bool& flag)
  {
    if (DataPM::mNelements == 0) {
      flag = false;
      return;
    }
    if (DataPM::mNelements % 2 == 0 && DataPM::mData[DataPM::mNelements - 1].channelID == 0) {
      DataPM::mNelements--; // in case of half GBT-word filling
    }
    // TODO, Descriptor checking, Channel range
  }
};

// standard data block from TCM
template <typename ConfigType, typename RawHeaderTCMtype, typename RawDataTCMtype>
class DataBlockTCM : public DataBlockBase<DataBlockTCM, ConfigType, RawHeaderTCMtype, RawDataTCMtype>
{
 public:
  DataBlockTCM() = default;
  DataBlockTCM(const DataBlockTCM&) = default;
  typedef RawHeaderTCMtype RawHeaderTCM;
  typedef RawDataTCMtype RawDataTCM;
  typedef DataBlockWrapper<ConfigType, RawHeaderTCM> HeaderTCM;
  typedef DataBlockWrapper<ConfigType, RawDataTCM> DataTCM;

  const std::size_t getNgbtWords() const { return HeaderTCM::mData[0].nGBTWords; }
  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    HeaderTCM::deserialize(srcBytes, HeaderTCM::MaxNwords, srcByteShift);
    DataTCM::deserialize(srcBytes, HeaderTCM::mData[0].nGBTWords, srcByteShift);
  }
  std::vector<char> serialize() const
  {
    std::size_t nBytes = HeaderTCM::MaxNwords * HeaderTCM::sSizeWord;
    nBytes += HeaderTCM::mData[0].nGBTWords * HeaderTCM::sSizeWord;
    std::vector<char> vecBytes(nBytes);
    std::size_t destBytes = 0;
    HeaderTCM::serialize(vecBytes, HeaderTCM::MaxNwords, destBytes);
    DataTCM::serialize(vecBytes, HeaderTCM::mData[0].nGBTWords, destBytes);
    return vecBytes;
  }
  // Custom sanity checking for current deserialized block
  //  put here code for raw data checking
  void sanityCheck(bool& flag)
  {
    // TODO, Descriptor checking
  }
};

// extended TCM mode, 1 TCMdata + 8 TCMdataExtendedstructs
template <typename ConfigType, typename RawHeaderTCMextType, typename RawDataTCMtype, typename RawDataTCMextType>
class DataBlockTCMext : public DataBlockBase<DataBlockTCMext, ConfigType, RawHeaderTCMextType, RawDataTCMtype, RawDataTCMextType>
{
 public:
  DataBlockTCMext() = default;
  DataBlockTCMext(const DataBlockTCMext&) = default;
  typedef RawHeaderTCMextType RawHeaderTCMext;
  typedef RawDataTCMtype RawDataTCM;
  typedef RawDataTCMextType RawDataTCMext;
  typedef DataBlockWrapper<ConfigType, RawHeaderTCMext> HeaderTCMext;
  typedef DataBlockWrapper<ConfigType, RawDataTCM> DataTCM;
  typedef DataBlockWrapper<ConfigType, RawDataTCMext> DataTCMext;

  const std::size_t getNgbtWords() const { return HeaderTCMext::mData[0].nGBTWords; }
  void deserialize(gsl::span<const uint8_t> srcBytes, size_t& srcByteShift)
  {
    HeaderTCMext::deserialize(srcBytes, HeaderTCMext::MaxNwords, srcByteShift);
    DataTCM::deserialize(srcBytes, DataTCM::MaxNwords, srcByteShift);
    DataTCMext::deserialize(srcBytes, HeaderTCMext::mData[0].nGBTWords - DataTCM::MaxNwords, srcByteShift);
  }

  std::vector<char> serialize() const
  {
    std::size_t nBytes = HeaderTCMext::MaxNwords * HeaderTCMext::sSizeWord;
    nBytes += HeaderTCMext::mData[0].nGBTWords * HeaderTCMext::sSizeWord;
    std::vector<char> vecBytes(nBytes);
    std::size_t destBytes = 0;
    HeaderTCMext::serialize(vecBytes, HeaderTCMext::MaxNwords, destBytes);
    DataTCM::serialize(vecBytes, DataTCM::MaxNwords, destBytes);
    DataTCMext::serialize(vecBytes, HeaderTCMext::mData[0].nGBTWords - DataTCM::MaxNwords, destBytes);
    return vecBytes;
  }
  // Custom sanity checking for current deserialized block
  //  put here code for raw data checking
  void sanityCheck(bool& flag)
  {

    // TODO, Descriptor checking
  }
};

} // namespace fit
} // namespace o2
#endif
