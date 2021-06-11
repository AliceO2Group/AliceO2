// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

#ifndef O2_ANALYSIS_JEDERIVED_H
#define O2_ANALYSIS_JEDERIVED_H

#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace jejet
{
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_COLUMN(Energy, energy, float);
DECLARE_SOA_COLUMN(Mass, mass, float);
DECLARE_SOA_COLUMN(Area, area, float);
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, [](float pt, float phi) { return pt * TMath::Cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, [](float pt, float phi) { return pt * TMath::Sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, [](float pt, float eta) { return pt * TMath::SinH(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(P, p, [](float pt, float eta) { return pt * TMath::CosH(eta); }); //absolute p
} // namespace jejet

DECLARE_SOA_TABLE(JEJets, "AOD", "JEJET",
                  o2::soa::Index<>,
                  jejet::Pt,
                  jejet::Eta,
                  jejet::Phi,
                  jejet::Energy,
                  jejet::Mass,
                  jejet::Area,
                  jejet::Px<jejet::Pt, jejet::Phi>,
                  jejet::Py<jejet::Pt, jejet::Phi>,
                  jejet::Pz<jejet::Pt, jejet::Eta>,
                  jejet::P<jejet::Pt, jejet::Eta>);

using JEJet = JEJets::iterator;

namespace jeconstituent
{
DECLARE_SOA_INDEX_COLUMN(JEJet, jejet);
DECLARE_SOA_COLUMN(Pt, pt, float);
DECLARE_SOA_COLUMN(Eta, eta, float);
DECLARE_SOA_COLUMN(Phi, phi, float);
DECLARE_SOA_DYNAMIC_COLUMN(Px, px, [](float pt, float phi) { return pt * TMath::Cos(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Py, py, [](float pt, float phi) { return pt * TMath::Sin(phi); });
DECLARE_SOA_DYNAMIC_COLUMN(Pz, pz, [](float pt, float eta) { return pt * TMath::SinH(eta); });
DECLARE_SOA_DYNAMIC_COLUMN(P, p, [](float pt, float eta) { return pt * TMath::CosH(eta); }); //absolute p
} // namespace jeconstituent

DECLARE_SOA_TABLE(JEConstituents, "AOD", "JECONSTITUENT", o2::soa::Index<>,
                  jeconstituent::JEJetId,
                  jeconstituent::Pt, jeconstituent::Eta, jeconstituent::Phi,
                  jeconstituent::Px<jeconstituent::Pt, jeconstituent::Phi>,
                  jeconstituent::Py<jeconstituent::Pt, jeconstituent::Phi>,
                  jeconstituent::Pz<jeconstituent::Pt, jeconstituent::Eta>,
                  jeconstituent::P<jeconstituent::Pt, jeconstituent::Eta>);
using JEConstituent = JEConstituents::iterator;
} // namespace o2::aod

#endif // O2_ANALYSIS_JEDERIVED_H
