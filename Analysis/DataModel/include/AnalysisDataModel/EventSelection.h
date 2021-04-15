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
#include "AnalysisCore/TriggerAliases.h"

namespace o2::aod
{
// collision-joinable event selection decisions
namespace evsel
{
// TODO bool arrays are not supported? Storing in int32 for the moment
DECLARE_SOA_COLUMN(Alias, alias, int32_t[kNaliases]);
DECLARE_SOA_COLUMN(BBT0A, bbT0A, bool);          // beam-beam time in T0A
DECLARE_SOA_COLUMN(BBT0C, bbT0C, bool);          // beam-beam time in T0C
DECLARE_SOA_COLUMN(BBV0A, bbV0A, bool);          // beam-beam time in V0A
DECLARE_SOA_COLUMN(BBV0C, bbV0C, bool);          // beam-beam time in V0C
DECLARE_SOA_COLUMN(BGV0A, bgV0A, bool);          // beam-gas time in V0A
DECLARE_SOA_COLUMN(BGV0C, bgV0C, bool);          // beam-gas time in V0C
DECLARE_SOA_COLUMN(BBZNA, bbZNA, bool);          // beam-beam time in ZNA
DECLARE_SOA_COLUMN(BBZNC, bbZNC, bool);          // beam-beam time in ZNC
DECLARE_SOA_COLUMN(BBFDA, bbFDA, bool);          // beam-beam time in FDA
DECLARE_SOA_COLUMN(BBFDC, bbFDC, bool);          // beam-beam time in FDC
DECLARE_SOA_COLUMN(BGFDA, bgFDA, bool);          // beam-gas time in FDA
DECLARE_SOA_COLUMN(BGFDC, bgFDC, bool);          // beam-gas time in FDC
DECLARE_SOA_COLUMN(FoundFT0, foundFT0, int64_t); // the nearest FT0 signal
DECLARE_SOA_DYNAMIC_COLUMN(SEL7, sel7, [](bool bbV0A, bool bbV0C, bool bbZNA, bool bbZNC) -> bool { return bbV0A && bbV0C && bbZNA && bbZNC; });
DECLARE_SOA_DYNAMIC_COLUMN(SEL8, sel8, [](bool bbT0A, bool bbT0C, bool bbZNA, bool bbZNC) -> bool { return bbT0A && bbT0C && bbZNA && bbZNC; });
} // namespace evsel
DECLARE_SOA_TABLE(EvSels, "AOD", "EVSEL",
                  evsel::Alias,
                  evsel::BBT0A, evsel::BBT0C,
                  evsel::BBV0A, evsel::BBV0C, evsel::BGV0A, evsel::BGV0C,
                  evsel::BBZNA, evsel::BBZNC,
                  evsel::BBFDA, evsel::BBFDC, evsel::BGFDA, evsel::BGFDC,
                  evsel::SEL7<evsel::BBV0A, evsel::BBV0C, evsel::BBZNA, evsel::BBZNC>,
                  evsel::SEL8<evsel::BBT0A, evsel::BBT0C, evsel::BBZNA, evsel::BBZNC>,
                  evsel::FoundFT0);
using EvSel = EvSels::iterator;

DECLARE_SOA_TABLE(BcSels, "AOD", "BCSEL",
                  evsel::Alias,
                  evsel::BBT0A, evsel::BBT0C,
                  evsel::BBV0A, evsel::BBV0C, evsel::BGV0A, evsel::BGV0C,
                  evsel::BBZNA, evsel::BBZNC,
                  evsel::BBFDA, evsel::BBFDC, evsel::BGFDA, evsel::BGFDC);
using BcSel = BcSels::iterator;
} // namespace o2::aod

#endif // O2_ANALYSIS_EVENTSELECTION_H_
