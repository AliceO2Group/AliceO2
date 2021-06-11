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
//file RawWriterFT0.h Raw writer class for FT0
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
//Normal TCM mode
using RawWriterFT0 = o2::fit::RawWriterFIT<DigitBlockFT0, DataBlockPM, DataBlockTCM>;
//Extended TCM mode
//using RawWriterFT0ext = o2::fit::RawWriterFIT<DigitBlockFT0, DataBlockPM, DataBlockTCMext>;
} // namespace ft0
} // namespace o2

#endif
