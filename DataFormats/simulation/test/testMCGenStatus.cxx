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

#define BOOST_TEST_MODULE Test MCGenStatus class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/ParticleStatus.h"
#include "SimulationDataFormat/MCGenStatus.h"
#include "TFile.h"
#include "TParticle.h"

using namespace o2;
using namespace o2::mcgenstatus;

BOOST_AUTO_TEST_CASE(MCGenStatus_test)
{
  // check basic properties
  const int status{91};

  // should not be flagged as encoded when the encoding is set
  MCGenStatusEncoding enc1(isEncodedValue);
  BOOST_CHECK(enc1.isEncoded != isEncodedValue);

  // check if everuthing correctly assigned
  MCGenStatusEncoding enc2(status, -status);
  BOOST_CHECK(enc2.isEncoded == isEncodedValue);
  BOOST_CHECK(enc2.hepmc == status);
  BOOST_CHECK(enc2.gen == -status);

  // check if helper functionality works as expected
  BOOST_CHECK(getHepMCStatusCode(enc2) == status);
  BOOST_CHECK(getGenStatusCode(enc2) == -status);
  BOOST_CHECK(getHepMCStatusCode(enc2) == getHepMCStatusCode(enc2.fullEncoding));
  BOOST_CHECK(getGenStatusCode(enc2) == getGenStatusCode(enc2.fullEncoding));

  // check default constructed MCTrack object's status code
  MCTrack track1;
  BOOST_CHECK(track1.getStatusCode().fullEncoding == 0);

  // check MCTrack object constructed from TParticle (primary)
  TParticle part2(22, MCGenStatusEncoding(status, -status).fullEncoding, 1, 2, 3, 4, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0);
  part2.SetBit(ParticleStatus::kPrimary);
  MCTrack track2(part2);
  BOOST_CHECK(getHepMCStatusCode(track2.getStatusCode()) == status);
  BOOST_CHECK(getGenStatusCode(track2.getStatusCode()) == -status);

  // make sure status codes survive serialising and deserialising
  {
    // serialize it
    TFile f("MCGenStatusOut.root", "RECREATE");
    f.WriteObject(&track2, "MCTrack");
    f.Close();
  }

  {
    MCTrack* intrack = nullptr;
    TFile f("MCGenStatusOut.root", "OPEN");
    f.GetObject("MCTrack", intrack);
    BOOST_CHECK(getHepMCStatusCode(intrack->getStatusCode()) == status);
    BOOST_CHECK(getGenStatusCode(intrack->getStatusCode()) == -status);
  }

  // check MCTrack object constructed from TParticle (secondary)
  MCTrack track3(TParticle(22, MCGenStatusEncoding(status, -status).fullEncoding, 1, 2, 3, 4, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0));
  // both must now be -1 since not encoced and hence only number is returned
  BOOST_CHECK(getHepMCStatusCode(track3.getStatusCode()) == -1);
  BOOST_CHECK(getGenStatusCode(track3.getStatusCode()) == -1);
}
