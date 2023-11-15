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
// Based on the FT0 file RawEventData.h
//  Alla.Maevskaya@cern.ch
//  with Artur.Furs
//
#ifndef ALICEO2_FV0_RAWEVENTDATA_H_
#define ALICEO2_FV0_RAWEVENTDATA_H_

#include "FV0Base/Constants.h"
#include "DataFormatsFIT/RawEventData.h"
#include <Framework/Logger.h>
#include "Rtypes.h"

namespace o2
{
namespace fv0
{
using EventHeader = o2::fit::EventHeader;
using EventData = o2::fit::EventData;
using TCMdata = o2::fit::TCMdata;
using TCMdataExtended = o2::fit::TCMdataExtended;
} // namespace fv0
} // namespace o2
#endif
