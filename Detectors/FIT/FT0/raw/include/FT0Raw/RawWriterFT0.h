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
// file RawWriterFT0.h Raw writer class for FT0
//
// Artur.Furs
// afurs@cern.ch

#ifndef ALICEO2_FIT_RAWWRITERFT0_H_
#define ALICEO2_FIT_RAWWRITERFT0_H_
#include "FT0Raw/DataBlockFT0.h"
#include "FT0Raw/DigitBlockFT0.h"
#include "FITRaw/RawWriterFIT.h"

namespace o2
{
namespace ft0
{
// Normal TCM mode
using RawWriterFT0 = o2::fit::RawWriterFIT<DigitBlockFT0, DataBlockPM, DataBlockTCM>;
using RawWriterFT0_padded = o2::fit::RawWriterFIT<DigitBlockFT0, DataBlockPM::DataBlockInvertedPadding_t, DataBlockTCM::DataBlockInvertedPadding_t>;

// Extended TCM mode
// using RawWriterFT0ext = o2::fit::RawWriterFIT<DigitBlockFT0, DataBlockPM, DataBlockTCMext>;
} // namespace ft0
} // namespace o2

#endif
