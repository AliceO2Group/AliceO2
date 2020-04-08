// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_MULTIPLICITY_H_
#define O2_ANALYSIS_MULTIPLICITY_H_

#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace mult
{
DECLARE_SOA_COLUMN(MultV0A, multV0A, float);
DECLARE_SOA_COLUMN(MultV0C, multV0C, float);
DECLARE_SOA_DYNAMIC_COLUMN(MultV0M, multV0M, [](float multV0A, float multV0C) -> float { return multV0A + multV0C; });
} // namespace mult
DECLARE_SOA_TABLE(Mults, "AOD", "MULT", mult::MultV0A, mult::MultV0C, mult::MultV0M<mult::MultV0A, mult::MultV0C>);
using Mult = Mults::iterator;
} // namespace o2::aod

#endif // O2_ANALYSIS_MULTIPLICITY_H_
