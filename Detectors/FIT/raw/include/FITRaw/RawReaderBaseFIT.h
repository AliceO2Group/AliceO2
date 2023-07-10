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
// file RawReaderFITBase.h Base class for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
// Main purpuse is to decode FIT data blocks and push them to DigitBlockFIT for proccess
// Base class only provides static linkID-moduleType conformity

#ifndef ALICEO2_FIT_RAWREADERBASEFIT_H_
#define ALICEO2_FIT_RAWREADERBASEFIT_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "FITRaw/RawReaderBase.h"
#include <CommonDataFormat/InteractionRecord.h>
#include "Headers/RAWDataHeader.h"
#include <DataFormatsFIT/RawDataMetric.h>
#include <Framework/Logger.h>

#include <gsl/span>

namespace o2
{
namespace fit
{

// Common raw reader for FIT
template <typename DigitBlockFITtype, typename DataBlockPMtype, typename DataBlockTCMtype>
class RawReaderBaseFIT : public RawReaderBase<DigitBlockFITtype, DataBlockPMtype, DataBlockTCMtype>
{
 public:
  typedef DigitBlockFITtype DigitBlockFIT_t;
  typedef typename DigitBlockFIT_t::LookupTable_t LookupTable_t;

  typedef std::conditional_t<DataBlockPMtype::sIsPadded, typename DataBlockPMtype::DataBlockInvertedPadding_t, DataBlockPMtype> DataBlockPM_t;
  typedef std::conditional_t<DataBlockTCMtype::sIsPadded, typename DataBlockTCMtype::DataBlockInvertedPadding_t, DataBlockTCMtype> DataBlockTCM_t;

  typedef std::conditional_t<DataBlockPMtype::sIsPadded, DataBlockPMtype, typename DataBlockPMtype::DataBlockInvertedPadding_t> DataBlockPMpadded_t;
  typedef std::conditional_t<DataBlockTCMtype::sIsPadded, DataBlockTCMtype, typename DataBlockTCMtype::DataBlockInvertedPadding_t> DataBlockTCMpadded_t;

  typedef RawReaderBase<DigitBlockFITtype, DataBlockPMtype, DataBlockTCMtype> RawReaderBase_t;
  RawReaderBaseFIT() = default;
  ~RawReaderBaseFIT() = default;
  void reserve(std::size_t nElements, std::size_t nElemMap = 0)
  {
    auto& vecDataBlocksPM = RawReaderBase_t::template getVecDataBlocks<DataBlockPM_t>();
    vecDataBlocksPM.reserve(nElements);
    auto& vecDataBlocksTCM = RawReaderBase_t::template getVecDataBlocks<DataBlockTCM_t>();
    vecDataBlocksTCM.reserve(nElements);
    // one need to reserve memory for map
    for (std::size_t iElem = 0; iElem < nElemMap; iElem++) {
      RawReaderBase_t::mMapDigits.emplace(o2::InteractionRecord(0, iElem), o2::InteractionRecord(0, iElem));
    }
    RawReaderBase_t::mMapDigits.clear();
  }
  // deserialize payload to raw data blocks and proccesss them to digits
  template <typename... T>
  void process(bool isPadded, gsl::span<const uint8_t> payload, uint16_t feeID, T&&... feeParameters)
  {
    if (LookupTable_t::Instance().isTCM(std::forward<T>(feeParameters)...)) {
      // TCM data proccessing
      if (isPadded) {
        RawReaderBase_t::template processBinaryData<DataBlockTCMpadded_t>(payload, feeID, std::forward<T>(feeParameters)...);
      } else {
        RawReaderBase_t::template processBinaryData<DataBlockTCM_t>(payload, feeID, std::forward<T>(feeParameters)...);
      }
    } else if (LookupTable_t::Instance().isPM(std::forward<T>(feeParameters)...)) {
      // PM data proccessing
      if (isPadded) {
        RawReaderBase_t::template processBinaryData<DataBlockPMpadded_t>(payload, feeID, std::forward<T>(feeParameters)...);
      } else {
        RawReaderBase_t::template processBinaryData<DataBlockPM_t>(payload, feeID, std::forward<T>(feeParameters)...);
      }
    } else {
      auto& metric = RawReaderBase_t::addMetric(feeID, std::forward<T>(feeParameters)..., false);
      metric.addStatusBit(RawDataMetric::EStatusBits::kWrongChannelMapping);
      LOG(error) << "Unregistered in ChannelMap link!";
      metric.print();
    }
  }
};
} // namespace fit
} // namespace o2

#endif
