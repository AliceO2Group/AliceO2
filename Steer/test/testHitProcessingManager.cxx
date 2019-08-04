// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test HitProcessingTest class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include "Steer/HitProcessingManager.h"
#include <TFile.h>
#include <TTree.h>
#include <string>

namespace o2
{
namespace steer
{

BOOST_AUTO_TEST_CASE(HitProcessingTest)
{
  // make some mockup sim files
  auto makefile = [](std::string name, int n) {
    TFile file(name.c_str(), "RECREATE");
    TTree tree("o2sim", "");
    tree.SetEntries(n);
    tree.Write();
    file.Close();
  };

  auto& mgr = o2::steer::HitProcessingManager::instance();
  makefile("o2sim_1.root", 4);      // 4 background events
  mgr.addInputFile("o2sim_1.root"); // add background file
  makefile("o2sim_2.root", 5);
  mgr.addInputFile("o2sim_2.root"); // add background file
  makefile("o2sim_s_1.root", 5);
  mgr.addInputSignalFile("o2sim_s_1.root", 1); // signals of type 1
  makefile("o2sim_s_2.root", 20);
  mgr.addInputSignalFile("o2sim_s_2.root", 2); // signals of type 2
  makefile("o2sim_s_3.root", 13);
  mgr.addInputSignalFile("o2sim_s_3.root", 1); // signals of type 1

  // setup run (without giving number of collision)
  mgr.setupRun(100);
}
} // namespace steer
} // namespace o2
