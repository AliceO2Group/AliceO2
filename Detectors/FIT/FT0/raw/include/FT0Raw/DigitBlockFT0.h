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
// file DigitBlockFT0.h class  for proccessing RAW data into Digits
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
// Normal data taking mode
using DigitBlockFT0 = o2::fit::DigitBlockFIT<o2::ft0::SingleLUT, o2::ft0::Digit, o2::ft0::ChannelData>;
// TCM extended data taking mode
using DigitBlockFT0ext = o2::fit::DigitBlockFIText<o2::ft0::SingleLUT, o2::ft0::Digit, o2::ft0::ChannelData, o2::ft0::TriggersExt>;
} // namespace ft0
} // namespace o2
#endif
