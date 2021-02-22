// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_LFDERIVED_H
#define O2_ANALYSIS_LFDERIVED_H

#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
DECLARE_SOA_TABLE(LFCollisions, "AOD", "LFCOLLISION", o2::soa::Index<>,
                  o2::aod::collision::PosZ);
using LFCollision = LFCollisions::iterator;

namespace lftrack
{
DECLARE_SOA_INDEX_COLUMN(LFCollision, lfCollision);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(P, p, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(TpcNSigmaEl, tpcNSigmaEl, float);
DECLARE_SOA_COLUMN(TpcNSigmaMu, tpcNSigmaMu, float);
DECLARE_SOA_COLUMN(TpcNSigmaPi, tpcNSigmaPi, float);
DECLARE_SOA_COLUMN(TpcNSigmaKa, tpcNSigmaKa, float);
DECLARE_SOA_COLUMN(TpcNSigmaPr, tpcNSigmaPr, float);
DECLARE_SOA_COLUMN(TpcNSigmaDe, tpcNSigmaDe, float);
DECLARE_SOA_COLUMN(TpcNSigmaTr, tpcNSigmaTr, float);
DECLARE_SOA_COLUMN(TpcNSigmaHe, tpcNSigmaHe, float);
DECLARE_SOA_COLUMN(TpcNSigmaAl, tpcNSigmaAl, float);
DECLARE_SOA_COLUMN(TofNSigmaEl, tofNSigmaEl, float);
DECLARE_SOA_COLUMN(TofNSigmaMu, tofNSigmaMu, float);
DECLARE_SOA_COLUMN(TofNSigmaPi, tofNSigmaPi, float);
DECLARE_SOA_COLUMN(TofNSigmaKa, tofNSigmaKa, float);
DECLARE_SOA_COLUMN(TofNSigmaPr, tofNSigmaPr, float);
DECLARE_SOA_COLUMN(TofNSigmaDe, tofNSigmaDe, float);
DECLARE_SOA_COLUMN(TofNSigmaTr, tofNSigmaTr, float);
DECLARE_SOA_COLUMN(TofNSigmaHe, tofNSigmaHe, float);
DECLARE_SOA_COLUMN(TofNSigmaAl, tofNSigmaAl, float);
} // namespace lftrack
DECLARE_SOA_TABLE(LFTracks, "AOD", "LFTRACK", o2::soa::Index<>,
                  lftrack::Pt, lftrack::P, lftrack::Eta,
                  lftrack::TpcNSigmaEl, lftrack::TpcNSigmaMu,
                  lftrack::TpcNSigmaPi, lftrack::TpcNSigmaKa,
                  lftrack::TpcNSigmaPr, lftrack::TpcNSigmaDe,
                  lftrack::TpcNSigmaTr, lftrack::TpcNSigmaHe,
                  lftrack::TpcNSigmaAl);
using LFTrack = LFTracks::iterator;

DECLARE_SOA_TABLE(LFNucleiTracks, "AOD", "LFNUCLEITRACK", o2::soa::Index<>,
                  lftrack::LFCollisionId,
                  lftrack::Pt, lftrack::P,
                  lftrack::Eta, lftrack::Phi,
                  lftrack::TpcNSigmaEl, lftrack::TpcNSigmaMu,
                  lftrack::TpcNSigmaPi, lftrack::TpcNSigmaKa,
                  lftrack::TpcNSigmaPr, lftrack::TpcNSigmaDe,
                  lftrack::TpcNSigmaTr, lftrack::TpcNSigmaHe,
                  lftrack::TpcNSigmaAl,
                  lftrack::TofNSigmaEl, lftrack::TofNSigmaMu,
                  lftrack::TofNSigmaPi, lftrack::TofNSigmaKa,
                  lftrack::TofNSigmaPr, lftrack::TofNSigmaDe,
                  lftrack::TofNSigmaTr, lftrack::TofNSigmaHe,
                  lftrack::TofNSigmaAl);
using LFNucleiTrack = LFNucleiTracks::iterator;
} // namespace o2::aod

#endif // O2_ANALYSIS_LFDERIVED_H
