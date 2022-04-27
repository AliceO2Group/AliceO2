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

/// \file DCSDPHints.h
/// \brief DCS data point configuration for the TPC
///
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef O2_TPC_DCSDPHints_H_
#define O2_TPC_DCSDPHints_H_

#include <vector>
#include <fmt/format.h>

#include "DetectorsDCS/DCSDataPointHint.h"

namespace o2::tpc::dcs
{

std::vector<o2::dcs::test::HintType> getTPCDCSDPHints(const int maxSectors = 17);

} // namespace o2::tpc::dcs
#endif
