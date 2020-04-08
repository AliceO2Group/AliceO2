// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_EVENTSELECTION_H_
#define O2_ANALYSIS_EVENTSELECTION_H_

#include "Framework/AnalysisDataModel.h"

// TODO read nAliases from the alias map?
#define nAliases 2

namespace o2::aod
{
namespace evsel
{
// TODO bool arrays are not supported? Storing in int32 for the moment
DECLARE_SOA_COLUMN(Alias, alias, int32_t[nAliases]);
DECLARE_SOA_COLUMN(BBV0A, bbV0A, bool); // beam-beam time in V0A
DECLARE_SOA_COLUMN(BBV0C, bbV0C, bool); // beam-beam time in V0C
DECLARE_SOA_COLUMN(BGV0A, bgV0A, bool); // beam-gas time in V0A
DECLARE_SOA_COLUMN(BGV0C, bgV0C, bool); // beam-gas time in V0C
DECLARE_SOA_COLUMN(BBZNA, bbZNA, bool); // beam-beam time in ZNA
DECLARE_SOA_COLUMN(BBZNC, bbZNC, bool); // beam-beam time in ZNC
DECLARE_SOA_DYNAMIC_COLUMN(SEL7, sel7, [](bool bbV0A, bool bbV0C, bool bbZNA, bool bbZNC) -> bool { return bbV0A && bbV0C && bbZNA && bbZNC; });
} // namespace evsel
DECLARE_SOA_TABLE(EvSels, "AOD", "EVSEL",
                  evsel::Alias,
                  evsel::BBV0A, evsel::BBV0C, evsel::BGV0A, evsel::BGV0C, evsel::BBZNA, evsel::BBZNC,
                  evsel::SEL7<evsel::BBV0A, evsel::BBV0C, evsel::BBZNA, evsel::BBZNC>);
using EvSel = EvSels::iterator;
} // namespace o2::aod

#endif // O2_ANALYSIS_EVENTSELECTION_H_
