// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ANALYSIS_HFCANDIDATESELECTIONTABLES_H_
#define O2_ANALYSIS_HFCANDIDATESELECTIONTABLES_H_

namespace o2::aod
{
namespace hf_selcandidate_d0
{
DECLARE_SOA_COLUMN(IsSelD0, isSelD0, int);
DECLARE_SOA_COLUMN(IsSelD0bar, isSelD0bar, int);
} // namespace hf_selcandidate_d0
DECLARE_SOA_TABLE(HFSelD0Candidate, "AOD", "HFSELD0CAND", hf_selcandidate_d0::IsSelD0, hf_selcandidate_d0::IsSelD0bar);
} // namespace o2::aod

namespace o2::aod
{
namespace hf_selcandidate_lc
{
DECLARE_SOA_COLUMN(IsSelLc, isSelLc, int);
} // namespace hf_selcandidate_lc
DECLARE_SOA_TABLE(HFSelLcCandidate, "AOD", "HFSELLCCAND", hf_selcandidate_lc::IsSelLc);
} // namespace o2::aod

#endif // O2_ANALYSIS_HFCANDIDATESELECTIONTABLES_H_
