// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_TRIGGER_H_
#define O2_ANALYSIS_TRIGGER_H_

#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace trigger
{
DECLARE_SOA_COLUMN(H2, hasH2, bool);
DECLARE_SOA_COLUMN(H3, hasH3, bool);
DECLARE_SOA_COLUMN(He3, hasHe3, bool);
DECLARE_SOA_COLUMN(He4, hasHe4, bool);

} // namespace trigger

DECLARE_SOA_TABLE(NucleiTriggers, "AOD", "Nuclei Triggers", trigger::H2, trigger::H3, trigger::He3, trigger::He4);

using NucleiTrigger = NucleiTriggers::iterator;
} // namespace o2::aod

#endif // O2_ANALYSIS_TRIGGER_H_
