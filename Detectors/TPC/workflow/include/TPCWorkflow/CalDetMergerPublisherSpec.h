// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_CalDetMergerPublisherSpec_H
#define O2_TPC_CalDetMergerPublisherSpec_H

/// @file   CalDetMergerPublisherSpec.h
/// @brief  TPC CalDet merger and CCDB publisher
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace tpc
{

o2::framework::DataProcessorSpec getCalDetMergerPublisherSpec(uint32_t lanes, bool skipCCDB, bool dumpAfterComplete = false);

} // namespace tpc
} // namespace o2

#endif
