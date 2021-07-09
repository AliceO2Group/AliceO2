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
//file RawReaderFDDBase.h Base class for RAW data reading
//
// Artur.Furs
// afurs@cern.ch
//
//Main purpuse is to decode FDD data blocks and push them to DigitBlockFDD for proccess
//Base class only provides static linkID-moduleType conformity

#ifndef ALICEO2_FIT_RAWREADERFDDBASE_H_
#define ALICEO2_FIT_RAWREADERFDDBASE_H_
#include "FDDRaw/DataBlockFDD.h"
#include "FDDRaw/DigitBlockFDD.h"
#include "FITRaw/RawReaderBaseFIT.h"

namespace o2
{
namespace fdd
{
//Normal TCM mode
using RawReaderFDDBaseNorm = o2::fit::RawReaderBaseFIT<DigitBlockFDD, DataBlockPM, DataBlockTCM>;
//Extended TCM mode
using RawReaderFDDBaseExt = o2::fit::RawReaderBaseFIT<DigitBlockFDDext, DataBlockPM, DataBlockTCMext>;
} // namespace fdd
} // namespace o2

#endif
