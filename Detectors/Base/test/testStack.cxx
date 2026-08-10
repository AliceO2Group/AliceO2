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

#define BOOST_TEST_MODULE Test MCStack class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DetectorsBase/Stack.h"
#include "TFile.h"
#include "TMCProcess.h"

using namespace o2;

// unit tests on MC stack
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

// convenience wrapper to push a track and return the assigned trackID
static int pushTrack(o2::data::Stack& st, int parentId, TMCProcess proc)
{
  int trackId;
  st.PushTrack(1, parentId, 0, 0., 0., 0., 10., 5., 5., 5., 0.1, 0., 0., 0., proc, trackId, 1., 1);
  return trackId;
}

// unit test for the radioactive-decay ancestry query
BOOST_AUTO_TEST_CASE(Stack_isFromRadDecay_test)
{
  o2::data::Stack st;

  // two primaries; note that primaries do not enter mParticles, only secondaries do
  const auto prim0 = pushTrack(st, -1, kPPrimary);
  const auto prim1 = pushTrack(st, -1, kPPrimary);

  // a radioactive decay product of the second primary, and its descendants.
  // this is deliberately the *first* secondary of the primary, so that it lands
  // in the first entry of the particle buffer
  const auto radDecay = pushTrack(st, prim1, kPRadDecay);
  const auto radChild = pushTrack(st, radDecay, kPHadronic);
  const auto radGrandChild = pushTrack(st, radChild, kPHadronic);

  // a plain secondary of the second primary: no radioactive decay anywhere in its history
  const auto ordinary = pushTrack(st, prim1, kPHadronic);

  // primaries can never come from a radioactive decay
  BOOST_CHECK(!st.isFromRadDecay(prim0));
  BOOST_CHECK(!st.isFromRadDecay(prim1));

  // a secondary whose ancestry ends in a primary must terminate the search with false
  BOOST_CHECK(!st.isFromRadDecay(ordinary));

  // directly and indirectly from a radioactive decay
  BOOST_CHECK(st.isFromRadDecay(radDecay));
  BOOST_CHECK(st.isFromRadDecay(radChild));
  BOOST_CHECK(st.isFromRadDecay(radGrandChild));

  // out-of-range track IDs are rejected rather than looked up
  BOOST_CHECK(!st.isFromRadDecay(-1));
  BOOST_CHECK(!st.isFromRadDecay(1000000000));
}
