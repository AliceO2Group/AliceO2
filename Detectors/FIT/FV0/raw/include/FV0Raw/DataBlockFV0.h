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
//file DataBlockFV0.h class  for RAW data format data blocks at FV0
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
//Raw event data for FV0
using RawHeaderPM = o2::fv0::EventHeader;
using RawDataPM = o2::fv0::EventData;
using RawHeaderTCM = o2::fv0::EventHeader;
using RawDataTCM = o2::fv0::TCMdata;
using RawHeaderTCMext = o2::fv0::EventHeader;
using RawDataTCMext = o2::fv0::TCMdataExtended;
//Data block for FV0 modules
using DataBlockPM = o2::fit::DataBlockPM<RawHeaderPM, RawDataPM>;
using DataBlockTCM = o2::fit::DataBlockTCM<RawHeaderTCM, RawDataTCM>;
using DataBlockTCMext = o2::fit::DataBlockTCMext<RawHeaderTCMext, RawDataTCM, RawDataTCMext>;
} // namespace fv0
} // namespace o2
#endif
