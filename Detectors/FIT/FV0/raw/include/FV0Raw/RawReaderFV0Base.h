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
//file RawReaderFV0Base.h Base class for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode FV0 data blocks and push them to DigitBlockFV0 for proccess
//Base class only provides static linkID-moduleType conformity

#ifndef ALICEO2_FIT_RAWREADERFV0BASE_H_
#define ALICEO2_FIT_RAWREADERFV0BASE_H_
#include "FV0Raw/DataBlockFV0.h"
#include "FV0Raw/DigitBlockFV0.h"
#include "FITRaw/RawReaderBaseFIT.h"

namespace o2
{
namespace fv0
{
//Normal TCM mode
using RawReaderFV0BaseNorm = o2::fit::RawReaderBaseFIT<DigitBlockFV0, DataBlockPM, DataBlockTCM>;
//Extended TCM mode
using RawReaderFV0BaseExt = o2::fit::RawReaderBaseFIT<DigitBlockFV0ext, DataBlockPM, DataBlockTCMext>;
} // namespace fv0
} // namespace o2

#endif
