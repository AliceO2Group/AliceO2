// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ANALYSIS_CANDIDATESELECTIONTABLES_H_
#define O2_ANALYSIS_CANDIDATESELECTIONTABLES_H_

namespace o2::aod
{
namespace hfselcandidate
{
DECLARE_SOA_COLUMN(IsSelD0, isSelD0, int);
DECLARE_SOA_COLUMN(IsSelD0bar, isSelD0bar, int);
} // namespace hfselcandidate
DECLARE_SOA_TABLE(HFSelD0Candidate, "AOD", "HFSELD0CAND", hfselcandidate::IsSelD0, hfselcandidate::IsSelD0bar);
} // namespace o2::aod

#endif // O2_ANALYSIS_CANDIDATESELECTIONTABLES_H_
