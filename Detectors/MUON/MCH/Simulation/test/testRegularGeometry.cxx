// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHSimulation RegularGeometry
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "DetectorsPassive/Absorber.h"
#include "DetectorsPassive/Cave.h"
#include "DetectorsPassive/Compensator.h"
#include "DetectorsPassive/Dipole.h"
#include "DetectorsPassive/Pipe.h"
#include "DetectorsPassive/Shil.h"
#include "MCHSimulation/Detector.h"
#include "TGeoManager.h"
#include <boost/test/data/test_case.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

BOOST_AUTO_TEST_CASE(DoNotThrow)
{
  if (gGeoManager && gGeoManager->GetTopVolume()) {
    std::cerr << "Can only call this function with an empty geometry, i.e. gGeoManager==nullptr "
              << " or gGeoManager->GetTopVolume()==nullptr\n";
  }
  TGeoManager* g = new TGeoManager("MCH-BASICS", "ALICE MCH Regular Geometry");
  o2::passive::Cave("CAVE", "Cave (for MCH Basics)").ConstructGeometry();
  o2::passive::Dipole("DIPO", "Alice Dipole (for MCH Basics)").ConstructGeometry();
  o2::passive::Compensator("COMP", "Alice Compensator Dipole (for MCH Basics)").ConstructGeometry();
  o2::passive::Pipe("PIPE", "Beam pipe (for MCH Basics)").ConstructGeometry();
  o2::passive::Shil("SHIL", "Small angle beam shield (for MCH Basics)").ConstructGeometry();
  o2::passive::Absorber("ABSO", "Absorber (for MCH Basics)").ConstructGeometry();
  BOOST_CHECK_NO_THROW((o2::mch::Detector(true).ConstructGeometry()));
}
