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
// file DataBlockFV0.h class  for RAW data format data blocks at FV0
//
// Artur.Furs
// afurs@cern.ch
//

#ifndef ALICEO2_FIT_DATABLOCKFV0_H_
#define ALICEO2_FIT_DATABLOCKFV0_H_
#include <DataFormatsFV0/RawEventData.h>
#include <FITRaw/DataBlockFIT.h>
namespace o2
{
namespace fv0
{
// Raw event data for FV0
using RawHeaderPM = o2::fv0::EventHeader;
using RawDataPM = o2::fv0::EventData;
using RawHeaderTCM = o2::fv0::EventHeader;
using RawDataTCM = o2::fv0::TCMdata;
using RawHeaderTCMext = o2::fv0::EventHeader;
using RawDataTCMext = o2::fv0::TCMdataExtended;
// Data block for FV0 modules, no padding - DataBlockConfig<false>
using DataBlockPM = o2::fit::DataBlockPM<o2::fit::DataBlockConfig<false>, RawHeaderPM, RawDataPM>;
using DataBlockTCM = o2::fit::DataBlockTCM<o2::fit::DataBlockConfig<false>, RawHeaderTCM, RawDataTCM>;
using DataBlockTCMext = o2::fit::DataBlockTCMext<o2::fit::DataBlockConfig<false>, RawHeaderTCMext, RawDataTCM, RawDataTCMext>;
} // namespace fv0
} // namespace o2
#endif
