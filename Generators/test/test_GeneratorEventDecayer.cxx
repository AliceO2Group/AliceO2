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

#define BOOST_TEST_MODULE Test decay step of Generator
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <Generators/GeneratorService.h>
#include <SimulationDataFormat/MCTrack.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/MCGenProperties.h>
#include <CommonUtils/ConfigurableParam.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace o2::eventgen;

namespace
{
// forces Xic+ -> Xi(1530)0 pi+ with the Geant4 lifetime
const std::string FORCE = "4232:mayDecay = on\n4232:onMode = off\n4232:oneChannel = 1 1.0 0 3324 211\n4232:tau0 = 0.1358\n";

std::string writeCfg(std::string const& text)
{
  char name[] = "/tmp/test_GeneratorEventDecayer_XXXXXX";
  int fd = mkstemp(name);
  close(fd);
  std::ofstream out(name);
  out << text;
  return name;
}

// generates one boxgen event of five Xic+ with the decay step on or off
std::pair<std::vector<o2::MCTrack>, o2::dataformats::MCEventHeader> generate(bool enable)
{
  static const std::string forcePath = writeCfg(FORCE);
  o2::conf::ConfigurableParam::updateFromString("BoxGun.pdg=4232;BoxGun.number=5;EventDecayer.config[1]=" + forcePath +
                                                ";EventDecayer.enable=" + (enable ? "true" : "false"));
  GeneratorService service;
  service.initService("boxgen", "", o2::eventgen::NoVertexOption());
  return service.generateEvent();
}
} // namespace

BOOST_AUTO_TEST_CASE(decay_step_on)
{
  auto [tracks, header] = generate(true);
  int nXic = 0;
  int nXiStar = 0;
  for (int i = 0; i < (int)tracks.size(); ++i) {
    auto const& t = tracks[i];
    if (t.GetPdgCode() == 4232) {
      ++nXic;
      BOOST_CHECK(!t.getToBeDone());
      BOOST_CHECK_EQUAL(o2::mcgenstatus::getHepMCStatusCode(t.getStatusCode()), 2);
    }
    if (t.GetPdgCode() == 3324) {
      ++nXiStar;
      int m = t.getMotherTrackId();
      BOOST_REQUIRE(m >= 0 && m < (int)tracks.size());
      BOOST_CHECK_EQUAL(tracks[m].GetPdgCode(), 4232);
      BOOST_CHECK(!t.getToBeDone());
      // the Xic+ decay point in cm, displaced from the origin by about its decay length
      double r = std::sqrt(t.Vx() * t.Vx() + t.Vy() * t.Vy() + t.Vz() * t.Vz());
      BOOST_CHECK(r > 0. && r < 10.);
      for (int d = t.getFirstDaughterTrackId(); d >= 0 && d <= t.getLastDaughterTrackId(); ++d) {
        BOOST_CHECK_EQUAL(tracks[d].getMotherTrackId(), i);
        BOOST_CHECK(tracks[d].getToBeDone());
      }
    }
  }
  BOOST_CHECK_EQUAL(nXic, 5);
  BOOST_CHECK_EQUAL(nXiStar, 5);
  bool valid = false;
  auto summary = header.getInfo<std::string>("eventDecayer", valid);
  BOOST_CHECK(valid);
  BOOST_CHECK(summary.find("pythia8/decays/base.cfg") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(decay_step_off)
{
  auto [tracks, header] = generate(false);
  BOOST_REQUIRE_EQUAL(tracks.size(), 5);
  for (auto const& t : tracks) {
    BOOST_CHECK_EQUAL(t.GetPdgCode(), 4232);
    BOOST_CHECK(t.getToBeDone());
    BOOST_CHECK_EQUAL(t.getFirstDaughterTrackId(), -1);
  }
  BOOST_CHECK(!header.hasInfo("eventDecayer"));
}
