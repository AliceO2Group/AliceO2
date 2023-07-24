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
// file DigitBlockFDD.h class  for proccessing RAW data into Digits
//
// Artur.Furs
// afurs@cern.ch

#ifndef ALICEO2_FIT_DIGITBLOCKFDD_H_
#define ALICEO2_FIT_DIGITBLOCKFDD_H_
#include "FITRaw/DigitBlockFIT.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/ChannelData.h"
#include "DataFormatsFDD/LookUpTable.h"

namespace o2
{
namespace fdd
{
// Normal data taking mode
using DigitBlockFDD = o2::fit::DigitBlockFIT<o2::fdd::SingleLUT, o2::fdd::Digit, o2::fdd::ChannelData>;
// TCM extended data taking mode
using DigitBlockFDDext = o2::fit::DigitBlockFIText<o2::fdd::SingleLUT, o2::fdd::Digit, o2::fdd::ChannelData, o2::fdd::TriggersExt>;
} // namespace fdd
} // namespace o2
#endif
