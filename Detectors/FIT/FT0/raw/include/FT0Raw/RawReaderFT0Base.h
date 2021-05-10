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
//file RawReaderFT0Base.h Base class for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode FT0 data blocks and push them to DigitBlockFT0 for proccess
//Base class only provides static linkID-moduleType conformity

#ifndef ALICEO2_FIT_RAWREADERFT0BASE_H_
#define ALICEO2_FIT_RAWREADERFT0BASE_H_
#include "FT0Raw/DataBlockFT0.h"
#include "FT0Raw/DigitBlockFT0.h"
#include "FITRaw/RawReaderBaseFIT.h"

namespace o2
{
namespace ft0
{
//Normal TCM mode
using RawReaderFT0BaseNorm = o2::fit::RawReaderBaseFIT<DigitBlockFT0, DataBlockPM, DataBlockTCM>;
//Extended TCM mode
using RawReaderFT0BaseExt = o2::fit::RawReaderBaseFIT<DigitBlockFT0ext, DataBlockPM, DataBlockTCMext>;
} // namespace ft0
} // namespace o2

#endif
