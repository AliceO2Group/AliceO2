// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// @author Marco Giacalone
/// co-authored by Claude Opus 5.5
/// @date 11-2026

#define BOOST_TEST_MODULE Test O2DatabasePDG
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "SimulationDataFormat/O2DatabasePDG.h"

using namespace o2;

// The nucleus remnant codes below are the ones Pythia8/Angantyr produces for heavy-ion runs:
// +-10LZZZAAA9, one lump per beam holding all the spectator nucleons of that beam.
BOOST_AUTO_TEST_CASE(NucleusRemnantCode_test)
{
  // shape of the code, both signs
  BOOST_CHECK(O2DatabasePDG::isNucleusRemnantCode(1000040089));
  BOOST_CHECK(O2DatabasePDG::isNucleusRemnantCode(-1000040089));
  BOOST_CHECK(O2DatabasePDG::isNucleusRemnantCode(1000822089));
  // ground states and other isomer levels are not remnants
  BOOST_CHECK(!O2DatabasePDG::isNucleusRemnantCode(1000040080));
  BOOST_CHECK(!O2DatabasePDG::isNucleusRemnantCode(1000040081));
  // a hypernucleus is not a remnant: Angantyr remnants carry no strangeness, and the mass
  // formula counts nucleons only, so it would miss the Lambda mass
  BOOST_CHECK(!O2DatabasePDG::isNucleusRemnantCode(1010010039));
  // not ions at all
  BOOST_CHECK(!O2DatabasePDG::isNucleusRemnantCode(211));
  BOOST_CHECK(!O2DatabasePDG::isNucleusRemnantCode(-211));
  BOOST_CHECK(!O2DatabasePDG::isNucleusRemnantCode(1100000009));

  int Z = -1, A = -1;
  O2DatabasePDG::nucleusRemnantZA(1000822089, Z, A);
  BOOST_CHECK_EQUAL(Z, 82);
  BOOST_CHECK_EQUAL(A, 208);
  O2DatabasePDG::nucleusRemnantZA(-1000000029, Z, A);
  BOOST_CHECK_EQUAL(Z, 0);
  BOOST_CHECK_EQUAL(A, 2);
  O2DatabasePDG::nucleusRemnantZA(211, Z, A);
  BOOST_CHECK_EQUAL(Z, 0);
  BOOST_CHECK_EQUAL(A, 0);
}

BOOST_AUTO_TEST_CASE(Mass_test)
{
  auto db = O2DatabasePDG::Instance();
  const double mp = db->GetParticle(2212)->Mass();
  const double mn = db->GetParticle(2112)->Mass();
  const double tolerance = 1.e-9;
  bool success = false;

  // ordinary particles and ions are looked up by their exact code
  BOOST_CHECK_CLOSE(O2DatabasePDG::Mass(211, success), db->GetParticle(211)->Mass(), tolerance);
  BOOST_CHECK(success);
  BOOST_CHECK_CLOSE(O2DatabasePDG::Mass(1000020030, success), db->GetParticle(1000020030)->Mass(), tolerance);
  BOOST_CHECK(success);

  // an isomere state falls back to its ground state
  BOOST_CHECK_CLOSE(O2DatabasePDG::Mass(1000020031, success), db->GetParticle(1000020030)->Mass(), tolerance);
  BOOST_CHECK(success);

  // so does an excited ion that Geant4 encoded with an unknown level (isomer digit 9)
  BOOST_CHECK_CLOSE(O2DatabasePDG::Mass(1000020049, success), db->GetParticle(1000020040)->Mass(), tolerance);
  BOOST_CHECK(success);

  // and so does a hypernucleus with that digit: the remnant formula would be short of the
  // Lambda mass, which is why isNucleusRemnantCode() requires L = 0
  BOOST_CHECK_CLOSE(O2DatabasePDG::Mass(1010010039, success), db->GetParticle(1010010030)->Mass(), tolerance);
  BOOST_CHECK(success);

  // a remnant whose ground state is known keeps that ground state mass, binding included
  BOOST_CHECK_CLOSE(O2DatabasePDG::Mass(1000010029, success), db->GetParticle(1000010020)->Mass(), tolerance);
  BOOST_CHECK(success);

  // a remnant that is no nucleus at all has no ground state: Z * m_p + (A - Z) * m_n
  BOOST_CHECK_CLOSE(O2DatabasePDG::Mass(1000000029, success), 2. * mn, tolerance);
  BOOST_CHECK(success);
  BOOST_CHECK_CLOSE(O2DatabasePDG::Mass(1000020029, success), 2. * mp, tolerance);
  BOOST_CHECK(success);
  BOOST_CHECK_CLOSE(O2DatabasePDG::Mass(1000010079, success), mp + 6. * mn, tolerance);
  BOOST_CHECK(success);

  // a nucleus and an anti-nucleus have the same mass, whichever branch they take
  for (int pdg : {1000822089, 1000010029, 1000000029, 1000040089, 1000080169, 1000020030}) {
    bool successAnti = false;
    const auto mass = O2DatabasePDG::Mass(pdg, success);
    const auto massAnti = O2DatabasePDG::Mass(-pdg, successAnti);
    BOOST_CHECK(success);
    BOOST_CHECK(successAnti);
    BOOST_CHECK_CLOSE(mass, massAnti, tolerance);
  }

  // unknown codes still fail, they do not get an invented mass
  BOOST_CHECK(O2DatabasePDG::Mass(99999999, success) < 0.);
  BOOST_CHECK(!success);
  BOOST_CHECK(O2DatabasePDG::Mass(1000990081, success) < 0.);
  BOOST_CHECK(!success);
  BOOST_CHECK(O2DatabasePDG::nucleusRemnantMass(211) < 0.);
}
