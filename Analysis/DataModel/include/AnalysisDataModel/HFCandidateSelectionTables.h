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
DECLARE_SOA_COLUMN(IsSelD0ToPiK, isSelD0ToPiK, int);
DECLARE_SOA_COLUMN(IsSelD0ToPiKbar, isSelD0ToPiKbar, int);
DECLARE_SOA_COLUMN(SelectionTopol, selectionTopol, int);
DECLARE_SOA_COLUMN(SelectionTopolConjugate, selectionTopolConjugate, int);
DECLARE_SOA_COLUMN(SelectionPID, selectionPID, int);
} // namespace hf_selcandidate_d0
DECLARE_SOA_TABLE(HFSelD0ToPiKCandidate, "AOD", "HFSELD0ToPiK", hf_selcandidate_d0::IsSelD0ToPiK, hf_selcandidate_d0::IsSelD0ToPiKbar);
DECLARE_SOA_TABLE(HFSelD0ToPiKCuts, "AOD", "HFCutD0ToPiK", hf_selcandidate_d0::SelectionTopol,
                  hf_selcandidate_d0::SelectionTopolConjugate, hf_selcandidate_d0::SelectionPID);
} // namespace o2::aod

namespace o2::aod
{
namespace hf_selcandidate_lc
{
DECLARE_SOA_COLUMN(IsSelLcToPKPi, isSelLcToPKPi, int);
DECLARE_SOA_COLUMN(IsSelLcToPiKP, isSelLcToPiKP, int);
DECLARE_SOA_COLUMN(SelectionTopol, selectionTopol, int);
DECLARE_SOA_COLUMN(SelectionTopolConjugate, selectionTopolConjugate, int);
DECLARE_SOA_COLUMN(SelectionPID, selectionPID, int);
} // namespace hf_selcandidate_lc
DECLARE_SOA_TABLE(HFSelLcToPKPiCandidate, "AOD", "HFSELLCToPKPi", hf_selcandidate_lc::IsSelLcToPKPi, hf_selcandidate_lc::IsSelLcToPiKP);
DECLARE_SOA_TABLE(HFSelLcToPKPiCuts, "AOD", "HFCutLCToPKPi", hf_selcandidate_lc::SelectionTopol,
                  hf_selcandidate_lc::SelectionTopolConjugate, hf_selcandidate_lc::SelectionPID);

} // namespace o2::aod

namespace o2::aod
{
namespace hf_selcandidate_jpsi
{
DECLARE_SOA_COLUMN(IsSelJpsiToEE, isSelJpsiToEE, int);
DECLARE_SOA_COLUMN(SelectionTopol, selectionTopol, int);
DECLARE_SOA_COLUMN(SelectionPID, selectionPID, int);
} // namespace hf_selcandidate_jpsi
DECLARE_SOA_TABLE(HFSelJpsiToEECandidate, "AOD", "HFSELJPSIToEE", hf_selcandidate_jpsi::IsSelJpsiToEE);
DECLARE_SOA_TABLE(HFSelJpsiToEECuts, "AOD", "HFCutJpsiToEE", hf_selcandidate_jpsi::SelectionTopol,
                  hf_selcandidate_jpsi::SelectionPID);
} // namespace o2::aod
#endif // O2_ANALYSIS_HFCANDIDATESELECTIONTABLES_H_
