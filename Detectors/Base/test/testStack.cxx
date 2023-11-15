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
