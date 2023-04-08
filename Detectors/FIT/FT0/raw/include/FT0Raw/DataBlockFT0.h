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
// file DataBlockFT0.h class  for RAW data format data blocks at FT0
//
// Artur.Furs
// afurs@cern.ch
//

#ifndef ALICEO2_FIT_DATABLOCKFT0_H_
#define ALICEO2_FIT_DATABLOCKFT0_H_
#include <DataFormatsFT0/RawEventData.h>
#include <FITRaw/DataBlockFIT.h>
namespace o2
{
namespace ft0
{
// Raw event data for FT0
using RawHeaderPM = o2::ft0::EventHeader;
using RawDataPM = o2::ft0::EventData;
using RawHeaderTCM = o2::ft0::EventHeader;
using RawDataTCM = o2::ft0::TCMdata;
using RawHeaderTCMext = o2::ft0::EventHeader;
using RawDataTCMext = o2::ft0::TCMdataExtended;
// Data block for FT0 modules, no padding - DataBlockConfig<false>
using DataBlockPM = o2::fit::DataBlockPM<o2::fit::DataBlockConfig<false>, RawHeaderPM, RawDataPM>;
using DataBlockTCM = o2::fit::DataBlockTCM<o2::fit::DataBlockConfig<false>, RawHeaderTCM, RawDataTCM>;
using DataBlockTCMext = o2::fit::DataBlockTCMext<o2::fit::DataBlockConfig<false>, RawHeaderTCMext, RawDataTCM, RawDataTCMext>;
} // namespace ft0
} // namespace o2
#endif
