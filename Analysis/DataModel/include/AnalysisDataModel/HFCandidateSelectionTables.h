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

#ifndef O2_ANALYSIS_HFCANDIDATESELECTIONTABLES_H_
#define O2_ANALYSIS_HFCANDIDATESELECTIONTABLES_H_

namespace o2::aod
{
namespace hf_selcandidate_d0
{
DECLARE_SOA_COLUMN(IsSelD0, isSelD0, int);       //!
DECLARE_SOA_COLUMN(IsSelD0bar, isSelD0bar, int); //!
} // namespace hf_selcandidate_d0
DECLARE_SOA_TABLE(HFSelD0Candidate, "AOD", "HFSELD0CAND", //!
                  hf_selcandidate_d0::IsSelD0, hf_selcandidate_d0::IsSelD0bar);

namespace hf_selcandidate_dplus
{
DECLARE_SOA_COLUMN(IsSelDplusToPiKPi, isSelDplusToPiKPi, int); //!
} // namespace hf_selcandidate_dplus
DECLARE_SOA_TABLE(HFSelDplusToPiKPiCandidate, "AOD", "HFSELDPLUSCAND", //!
                  hf_selcandidate_dplus::IsSelDplusToPiKPi);

namespace hf_selcandidate_lc
{
DECLARE_SOA_COLUMN(IsSelLcpKpi, isSelLcpKpi, int); //!
DECLARE_SOA_COLUMN(IsSelLcpiKp, isSelLcpiKp, int); //!
} // namespace hf_selcandidate_lc
DECLARE_SOA_TABLE(HFSelLcCandidate, "AOD", "HFSELLCCAND", //!
                  hf_selcandidate_lc::IsSelLcpKpi, hf_selcandidate_lc::IsSelLcpiKp);

namespace hf_selcandidate_jpsi
{
DECLARE_SOA_COLUMN(IsSelJpsiToEE, isSelJpsiToEE, int); //!
} // namespace hf_selcandidate_jpsi
DECLARE_SOA_TABLE(HFSelJpsiToEECandidate, "AOD", "HFSELJPSICAND", //!
                  hf_selcandidate_jpsi::IsSelJpsiToEE);
namespace hf_selcandidate_lc_k0sp
{
DECLARE_SOA_COLUMN(IsSelLcK0sP, isSelLcK0sP, int);
} // namespace hf_selcandidate_lc_k0sp
DECLARE_SOA_TABLE(HFSelLcK0sPCandidate, "AOD", "HFSELLCK0SPCAND", //!
                  hf_selcandidate_lc_k0sp::IsSelLcK0sP);

} // namespace o2::aod

namespace o2::aod
{
namespace hf_selcandidate_xic
{
DECLARE_SOA_COLUMN(IsSelXicToPKPi, isSelXicToPKPi, int); //!
DECLARE_SOA_COLUMN(IsSelXicToPiKP, isSelXicToPiKP, int); //!
} // namespace hf_selcandidate_xic
DECLARE_SOA_TABLE(HFSelXicToPKPiCandidate, "AOD", "HFSELXICCAND", //!
                  hf_selcandidate_xic::IsSelXicToPKPi, hf_selcandidate_xic::IsSelXicToPiKP);
} // namespace o2::aod
#endif // O2_ANALYSIS_HFCANDIDATESELECTIONTABLES_H_
