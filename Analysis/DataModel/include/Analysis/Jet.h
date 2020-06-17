// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// table definitions for jets
//
// Author: Jochen Klein

#pragma once

#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace jet
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(Area, area, float);
DECLARE_SOA_COLUMN(Energy, energy, float);
DECLARE_SOA_COLUMN(Mass, mass, float);
} // namespace jet

DECLARE_SOA_TABLE(Jets, "AOD", "JET",
                  o2::soa::Index<>, jet::CollisionId,
                  jet::Eta, jet::Phi, jet::Pt, jet::Area, jet::Energy, jet::Mass);

using Jet = Jets::iterator;

// TODO: absorb in jet table
// when list of references available
namespace constituents
{
DECLARE_SOA_INDEX_COLUMN(Jet, jet);
DECLARE_SOA_INDEX_COLUMN(Track, track);
} // namespace constituents

DECLARE_SOA_TABLE(JetConstituents, "AOD", "JETCONSTITUENTS",
                  constituents::JetId, constituents::TrackId);

using JetConstituent = JetConstituents::iterator;
} // namespace o2::aod
