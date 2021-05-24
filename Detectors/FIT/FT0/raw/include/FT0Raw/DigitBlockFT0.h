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
//file DigitBlockFT0.h class  for proccessing RAW data into Digits
//
// Artur.Furs
// afurs@cern.ch

#ifndef ALICEO2_FIT_DIGITBLOCKFT0_H_
#define ALICEO2_FIT_DIGITBLOCKFT0_H_
#include "FITRaw/DigitBlockFIT.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/LookUpTable.h"

namespace o2
{
namespace ft0
{
//Normal data taking mode
using DigitBlockFT0 = DigitBlockFIT<o2::ft0::SingleLUT, o2::ft0::Digit, o2::ft0::ChannelData>;
//TCM extended data taking mode
using DigitBlockFT0ext = DigitBlockFIText<o2::ft0::SingleLUT, o2::ft0::Digit, o2::ft0::ChannelData, o2::ft0::TriggersExt>;
} // namespace ft0
} // namespace o2
#endif
