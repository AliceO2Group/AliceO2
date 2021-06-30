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
#ifndef O2_ANALYSIS_CENTRALITY_H_
#define O2_ANALYSIS_CENTRALITY_H_

#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace cent
{
DECLARE_SOA_COLUMN(CentV0M, centV0M, float); //!
} // namespace cent
DECLARE_SOA_TABLE(Cents, "AOD", "CENT", cent::CentV0M); //!
using Cent = Cents::iterator;
} // namespace o2::aod

#endif // O2_ANALYSIS_CENTRALITY_H_
