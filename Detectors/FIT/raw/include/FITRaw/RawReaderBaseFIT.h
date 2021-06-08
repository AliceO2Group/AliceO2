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
//file RawReaderFITBase.h Base class for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode FIT data blocks and push them to DigitBlockFIT for proccess
//Base class only provides static linkID-moduleType conformity

#ifndef ALICEO2_FIT_RAWREADERBASEFIT_H_
#define ALICEO2_FIT_RAWREADERBASEFIT_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "FITRaw/RawReaderBase.h"

#include "Headers/RAWDataHeader.h"

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
  typedef DataBlockPMtype DataBlockPM_t;
  typedef DataBlockTCMtype DataBlockTCM_t;
  typedef RawReaderBase<DigitBlockFIT_t, DataBlockPM_t, DataBlockTCM_t> RawReaderBase_t;
  RawReaderBaseFIT() = default;
  ~RawReaderBaseFIT() = default;
  //deserialize payload to raw data blocks and proccesss them to digits
  template <typename... T>
  void process(gsl::span<const uint8_t> payload, T&&... feeParameters)
  {
    if (LookupTable_t::Instance().isTCM(std::forward<T>(feeParameters)...)) {
      //TCM data proccessing
      RawReaderBase_t::template processBinaryData<DataBlockTCM_t>(payload, std::forward<T>(feeParameters)...);
    } else {
      //PM data proccessing
      RawReaderBase_t::template processBinaryData<DataBlockPM_t>(payload, std::forward<T>(feeParameters)...);
    }
  }
};
} // namespace fit
} // namespace o2

#endif
