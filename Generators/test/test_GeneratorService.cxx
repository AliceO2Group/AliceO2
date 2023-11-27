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

#define BOOST_TEST_MODULE Test GeneratorService class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <Generators/GeneratorService.h>
#include <SimulationDataFormat/MCTrack.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <CommonUtils/ConfigurableParam.h>

using Key = o2::dataformats::MCInfoKeys;
using namespace o2::eventgen;

BOOST_AUTO_TEST_CASE(boxgen_novertex)
{
  // a simple box generator without any vertex smearing applied
  GeneratorService service;
  service.initService("boxgen", "", o2::eventgen::NoVertexOption());
  auto event = service.generateEvent();

  BOOST_CHECK(event.first.size() > 0);

  auto& header = event.second;
  BOOST_CHECK(header.GetX() == 0);
  BOOST_CHECK(header.GetY() == 0);
  BOOST_CHECK(header.GetZ() == 0);
  BOOST_CHECK(header.GetT() == 0);
}

BOOST_AUTO_TEST_CASE(pythia8pp_diamonvertex)
{
  // parameter to influence vertex position
  o2::conf::ConfigurableParam::updateFromString("Diamond.position[0]=1.1; Diamond.width[0]=0.");

  GeneratorService service;
  // a pythia8 generator with vertex smearing coming from Diamond param
  service.initService("pythia8pp", "", o2::eventgen::DiamondParamVertexOption());
  auto event = service.generateEvent();

  BOOST_CHECK(event.first.size() > 0);

  auto& header = event.second;
  // BOOST_CHECK_CLOSE(header.GetX(), 1.1, 1E-6);
  // BOOST_CHECK(header.GetX() == 1.1);
  BOOST_CHECK(header.GetY() != 0.);
  BOOST_CHECK(header.GetZ() != 0.);
  BOOST_CHECK(header.GetT() == 0);

  bool invalid = false;
  BOOST_CHECK(header.getInfo<std::string>(Key::generator, invalid) == "pythia8");
}