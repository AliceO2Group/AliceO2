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
//file DigitBlockFV0.h class  for proccessing RAW data into Digits
//
// Artur.Furs
// afurs@cern.ch

#ifndef ALICEO2_FIT_DIGITBLOCKFV0_H_
#define ALICEO2_FIT_DIGITBLOCKFV0_H_
#include "FITRaw/DigitBlockFIT.h"
//#include "DataFormatsFV0/Digit.h"
#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/LookUpTable.h"

namespace o2
{
namespace fv0
{
//Normal data taking mode
//using DigitBlockFV0 = DigitBlockFIT<o2::fv0::SingleLUT, o2::fv0::Digit, o2::fv0::ChannelData>;
using DigitBlockFV0 = DigitBlockFIT<o2::fv0::SingleLUT, o2::fv0::BCData, o2::fv0::ChannelData>;
//TCM extended data taking mode
//using DigitBlockFV0ext = DigitBlockFIText<o2::fv0::SingleLUT, o2::fv0::Digit, o2::fv0::ChannelData, o2::fv0::TriggersExt>;
using DigitBlockFV0ext = DigitBlockFIText<o2::fv0::SingleLUT, o2::fv0::BCData, o2::fv0::ChannelData, o2::fv0::TriggersExt>;
} // namespace fv0
} // namespace o2
#endif
