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
//file RawWriterFDD.h Raw writer class for FDD
//
// Artur.Furs
// afurs@cern.ch

#ifndef ALICEO2_FIT_RAWWRITERFDD_H_
#define ALICEO2_FIT_RAWWRITERFDD_H_
#include "FDDRaw/DataBlockFDD.h"
#include "FDDRaw/DigitBlockFDD.h"
#include "FITRaw/RawWriterFIT.h"

namespace o2
{
namespace fdd
{
//Normal TCM mode
using RawWriterFDD = o2::fit::RawWriterFIT<DigitBlockFDD, DataBlockPM, DataBlockTCM>;
//Extended TCM mode
//using RawWriterFDDext = o2::fit::RawWriterFIT<DigitBlockFDD, DataBlockPM, DataBlockTCMext>;
} // namespace fdd
} // namespace o2

#endif
