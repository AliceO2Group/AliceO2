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
/// \file   MID.h
/// \author Vít Kučera
/// \note   Based on RICH.h
/// \brief  Set of tables for the ALICE3 MID information
///

#ifndef O2_ANALYSIS_ALICE3_MID_H_
#define O2_ANALYSIS_ALICE3_MID_H_

// O2 includes
#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace alice3mid
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);    //!
DECLARE_SOA_INDEX_COLUMN(Track, track);            //!
DECLARE_SOA_COLUMN(MIDIsMuon, midIsMuon, uint8_t); //! FIXME: To be changed to bool once bool columns are groupable.
} // namespace alice3mid

DECLARE_SOA_TABLE(MIDs, "AOD", "MID", //!
                  o2::soa::Index<>,
                  alice3mid::CollisionId,
                  alice3mid::TrackId,
                  alice3mid::MIDIsMuon);

using MID = MIDs::iterator;

} // namespace o2::aod

#endif // O2_ANALYSIS_ALICE3_MID_H_
