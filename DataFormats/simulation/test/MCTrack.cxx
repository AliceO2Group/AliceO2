// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCTrack class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include "SimulationDataFormat/MCTrack.h"
#include <boost/test/unit_test.hpp>
#include <iomanip>
#include <ios>
#include <iostream>
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/PrimaryChunk.h"
#include "TFile.h"
#include "TParticle.h"
#include "TMCProcess.h"

using namespace o2;

BOOST_AUTO_TEST_CASE(MCTrack_test)
{
  MCTrack track;
  // check properties on default constructed object
  BOOST_CHECK(track.getStore() == false);
  for (auto i = o2::detectors::DetID::First; i < o2::detectors::DetID::nDetectors; ++i) {
    BOOST_CHECK(track.leftTrace(i) == false);
  }
  BOOST_CHECK(track.getNumDet() == 0);
  BOOST_CHECK(track.hasHits() == false);

  // check storing
  track.setStore(true);
  BOOST_CHECK(track.getStore() == true);
  track.setStore(false);
  BOOST_CHECK(track.getStore() == false);
  track.setStore(true);
  BOOST_CHECK(track.getStore() == true);

  // set hit for first detector
  BOOST_CHECK(track.leftTrace(1) == false);
  track.setHit(1);
  BOOST_CHECK(track.hasHits() == true);
  BOOST_CHECK(track.leftTrace(1) == true);
  BOOST_CHECK(track.getNumDet() == 1);

  {
    // serialize it
    TFile f("MCTrackOut.root", "RECREATE");
    f.WriteObject(&track, "MCTrack");
    f.Close();
  }

  {
    MCTrack* intrack = nullptr;
    TFile f("MCTrackOut.root", "OPEN");
    f.GetObject("MCTrack", intrack);
    BOOST_CHECK(intrack->getStore() == true);
    BOOST_CHECK(intrack->hasHits() == true);
  }
}

// unit tests on stack
BOOST_AUTO_TEST_CASE(Stack_test)
{
  o2::data::Stack st;
  int a;
  TMCProcess proc{kPPrimary};
  // add a 2 primary particles
  st.PushTrack(1, -1, 0, 0, 0., 0., 10., 5., 5., 5., 0.1, 0., 0., 0., proc, a, 1., 1);
  st.PushTrack(1, -1, 0, 0, 0., 0., 10., 5., 5., 5., 0.1, 0., 0., 0., proc, a, 1., 1);
  BOOST_CHECK(st.getPrimaries().size() == 2);

  {
    // serialize it
    TFile f("StackOut.root", "RECREATE");
    f.WriteObject(&st, "Stack");
    f.Close();
  }

  {
    o2::data::Stack* inst = nullptr;
    TFile f("StackOut.root", "OPEN");
    f.GetObject("Stack", inst);
    BOOST_CHECK(inst->getPrimaries().size() == 2);
  }
}
