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
//file RawWriterFV0.h Raw writer class for FV0
//
// Artur.Furs
// afurs@cern.ch

#ifndef ALICEO2_FIT_RAWWRITERFV0_H_
#define ALICEO2_FIT_RAWWRITERFV0_H_
#include "FV0Raw/DataBlockFV0.h"
#include "FV0Raw/DigitBlockFV0.h"
#include "FITRaw/RawWriterFIT.h"

namespace o2
{
namespace fv0
{
//Normal TCM mode
using RawWriterFV0 = o2::fit::RawWriterFIT<DigitBlockFV0, DataBlockPM, DataBlockTCM>;
//Extended TCM mode
//using RawWriterFV0ext = o2::fit::RawWriterFIT<DigitBlockFV0, DataBlockPM, DataBlockTCMext>;
} // namespace fv0
} // namespace o2

#endif
