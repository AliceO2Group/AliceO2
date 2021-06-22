// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   RICH.h
/// \author Nicolo' Jacazio
/// \since  25/02/2021
/// \brief  Set of tables for the ALICE3 RICH information
///

#ifndef O2_ANALYSIS_ALICE3_RICH_H_
#define O2_ANALYSIS_ALICE3_RICH_H_

// O2 includes
#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace alice3rich
{
DECLARE_SOA_INDEX_COLUMN(Track, track);                      //!
DECLARE_SOA_COLUMN(RICHSignal, richSignal, float);           //!
DECLARE_SOA_COLUMN(RICHSignalError, richSignalError, float); //!
DECLARE_SOA_COLUMN(RICHDeltaEl, richDeltaEl, float);         //!
DECLARE_SOA_COLUMN(RICHDeltaMu, richDeltaMu, float);         //!
DECLARE_SOA_COLUMN(RICHDeltaPi, richDeltaPi, float);         //!
DECLARE_SOA_COLUMN(RICHDeltaKa, richDeltaKa, float);         //!
DECLARE_SOA_COLUMN(RICHDeltaPr, richDeltaPr, float);         //!
DECLARE_SOA_COLUMN(RICHNsigmaEl, richNsigmaEl, float);       //!
DECLARE_SOA_COLUMN(RICHNsigmaMu, richNsigmaMu, float);       //!
DECLARE_SOA_COLUMN(RICHNsigmaPi, richNsigmaPi, float);       //!
DECLARE_SOA_COLUMN(RICHNsigmaKa, richNsigmaKa, float);       //!
DECLARE_SOA_COLUMN(RICHNsigmaPr, richNsigmaPr, float);       //!
} // namespace alice3rich

DECLARE_SOA_TABLE(RICHs, "AOD", "RICH", //!
                  o2::soa::Index<>,
                  alice3rich::TrackId,
                  alice3rich::RICHSignal,
                  alice3rich::RICHSignalError,
                  alice3rich::RICHDeltaEl,
                  alice3rich::RICHDeltaMu,
                  alice3rich::RICHDeltaPi,
                  alice3rich::RICHDeltaKa,
                  alice3rich::RICHDeltaPr,
                  alice3rich::RICHNsigmaEl,
                  alice3rich::RICHNsigmaMu,
                  alice3rich::RICHNsigmaPi,
                  alice3rich::RICHNsigmaKa,
                  alice3rich::RICHNsigmaPr);

using RICH = RICHs::iterator;

} // namespace o2::aod

#endif // O2_ANALYSIS_ALICE3_RICH_H_
