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
//file RawReaderFDDBase.h Base class for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode FDD data blocks and push them to DigitBlockFDD for proccess
//Base class only provides static linkID-moduleType conformity

#ifndef ALICEO2_FIT_RAWREADERFDDBASE_H_
#define ALICEO2_FIT_RAWREADERFDDBASE_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "FDDRaw/DataBlockFDD.h"
#include "FDDRaw/DigitBlockFDD.h"
#include "FITRaw/RawReaderBase.h"

#include <boost/mpl/inherit.hpp>
#include <boost/mpl/vector.hpp>

#include <CommonDataFormat/InteractionRecord.h>
#include "Headers/RAWDataHeader.h"

#include <gsl/span>

using namespace o2::fit;
namespace o2
{
namespace fdd
{

// Common raw reader for FDD
template <class DigitBlockFDDtype, class DataBlockPMtype, class DataBlockTCMtype>
class RawReaderFDDBase : public RawReaderBase<DigitBlockFDDtype>
{
 public:
  typedef RawReaderBase<DigitBlockFDDtype> RawReaderBaseType;
  RawReaderFDDBase() = default;
  ~RawReaderFDDBase() = default;
  //deserialize payload to raw data blocks and proccesss them to digits
  void process(int linkID, gsl::span<const uint8_t> payload)
  {
    if (0 <= linkID && linkID < 2) {
      //PM data proccessing
      RawReaderBaseType::template processBinaryData<DataBlockPMtype>(payload, linkID);
    } else if (linkID == 2) {
      //TCM data proccessing
      RawReaderBaseType::template processBinaryData<DataBlockTCMtype>(payload, linkID);
    } else {
      //put here code in case of bad rdh.linkID value
      LOG(INFO) << "WARNING! WRONG LINK ID! " << linkID;
      return;
    }

    //
  }
};
//Normal TCM mode
using RawReaderFDDBaseNorm = RawReaderFDDBase<DigitBlockFDD, DataBlockPM, DataBlockTCM>;

} // namespace fdd
} // namespace o2

#endif
