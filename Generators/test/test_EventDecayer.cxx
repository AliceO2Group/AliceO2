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

#define BOOST_TEST_MODULE Test EventDecayer class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <Generators/EventDecayer.h>
#include <Generators/EventDecayerParam.h>
#include <SimulationDataFormat/MCUtils.h>
#include <SimulationDataFormat/MCGenProperties.h>
#include <SimulationDataFormat/ParticleStatus.h>
#include <SimulationDataFormat/O2DatabasePDG.h>
#include <CommonUtils/ConfigurableParam.h>
#include <TParticle.h>
#include <TRandom.h>
#include <TMCProcess.h>
#include <cmath>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace o2::eventgen;

namespace
{

// forces Xic+ -> Xi(1530)0 pi+ with the Geant4 lifetime
const std::string FORCE = "4232:mayDecay = on\n4232:onMode = off\n4232:oneChannel = 1 1.0 0 3324 211\n4232:tau0 = 0.1358\n";

// the default of EventDecayer.skipPdg
const std::string DEFAULTSKIP = "115 225 335 10221 10331 20223 20333 102132 102134 9010221";

// writes text to a fresh temporary file and returns its path
std::string writeCfg(std::string const& text)
{
  char name[] = "/tmp/test_EventDecayer_XXXXXX";
  int fd = mkstemp(name);
  close(fd);
  std::ofstream out(name);
  out << text;
  return name;
}

// a primary generator particle moving along z
TParticle makeParticle(int pdg, double p, bool toBeDone, double vx = 0, double vy = 0, double vz = 0)
{
  bool ok = false;
  double m = o2::O2DatabasePDG::Mass(pdg, ok);
  double e = std::sqrt(p * p + m * m);
  TParticle part(pdg, 1, -1, -1, -1, -1, 0., 0., p, e, vx, vy, vz, 0.);
  o2::mcutils::MCGenHelper::encodeParticleStatusAndTracking(part, 1, 1, toBeDone);
  part.SetUniqueID(kPPrimary);
  return part;
}

int hepmc(TParticle const& p)
{
  return o2::mcgenstatus::getHepMCStatusCode(p.GetStatusCode());
}

bool toBeDone(TParticle const& p)
{
  return p.TestBit(ParticleStatus::kToBeDone);
}

// resets the parameter to the forcing configuration with nothing else set
void setParams(std::string const& g4table = "", std::string const& extraCfg = "", int seed = 0)
{
  static const std::string forcePath = writeCfg(FORCE);
  o2::conf::ConfigurableParam::setValue("EventDecayer.config[0]", "");
  o2::conf::ConfigurableParam::setValue("EventDecayer.config[1]", forcePath);
  o2::conf::ConfigurableParam::setValue("EventDecayer.config[2]", extraCfg);
  o2::conf::ConfigurableParam::setValue("EventDecayer.g4ParticleTable", g4table);
  o2::conf::ConfigurableParam::setValue("EventDecayer.seed", std::to_string(seed));
  o2::conf::ConfigurableParam::setValue("EventDecayer.tau0Max", "10.");
  o2::conf::ConfigurableParam::setValue("EventDecayer.skipPdg", DEFAULTSKIP);
}

// the output of decaying one Xic+ at (0.1,0.2,0.3) cm with p = 5 GeV
std::vector<TParticle> decayOneXic(EventDecayerReport& report)
{
  setParams();
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  auto xic = makeParticle(4232, 5., true, 0.1, 0.2, 0.3);
  // a distinctive generator status and a non-zero start time
  o2::mcutils::MCGenHelper::encodeParticleStatusAndTracking(xic, 1, 85, true);
  xic.SetProductionVertex(0.1, 0.2, 0.3, 1e-9);
  std::vector<TParticle> particles{xic};
  report = decayer.decay(particles);
  return particles;
}

bool sameParticle(TParticle const& a, TParticle const& b)
{
  return a.GetPdgCode() == b.GetPdgCode() && a.Px() == b.Px() && a.Py() == b.Py() && a.Pz() == b.Pz() &&
         a.Energy() == b.Energy() && a.Vx() == b.Vx() && a.Vy() == b.Vy() && a.Vz() == b.Vz() && a.T() == b.T() &&
         a.GetStatusCode() == b.GetStatusCode() && a.TestBits(0xffffffff) == b.TestBits(0xffffffff) &&
         a.GetFirstMother() == b.GetFirstMother() && a.GetFirstDaughter() == b.GetFirstDaughter() &&
         a.GetLastDaughter() == b.GetLastDaughter() && a.GetUniqueID() == b.GetUniqueID();
}

// checks that a decay call left the particles as they were
void checkUnchanged(std::vector<TParticle> const& particles, std::vector<TParticle> const& before)
{
  BOOST_REQUIRE_EQUAL(particles.size(), before.size());
  for (size_t i = 0; i < before.size(); ++i) {
    BOOST_CHECK(sameParticle(particles[i], before[i]));
  }
}

// a Xi- or Xi0, or their antiparticles
bool isXi(TParticle const& p)
{
  return std::abs(p.GetPdgCode()) == 3312 || std::abs(p.GetPdgCode()) == 3322;
}

// 20000 Xic+ at p = 3 GeV decayed by one decayer, shared by the statistics cases
std::vector<TParticle> const& manyXic()
{
  static std::vector<TParticle> particles;
  if (particles.empty()) {
    setParams();
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    for (int i = 0; i < 20000; ++i) {
      particles.push_back(makeParticle(4232, 3., true));
    }
    auto report = decayer.decay(particles);
    BOOST_REQUIRE(report.decayed[4232] == 20000);
  }
  return particles;
}

} // namespace

BOOST_AUTO_TEST_CASE(xic_to_xistar_chain)
{
  EventDecayerReport report;
  auto particles = decayOneXic(report);

  auto const& xic = particles[0];
  BOOST_CHECK(!toBeDone(xic));
  BOOST_CHECK_EQUAL(hepmc(xic), 2);
  BOOST_CHECK_EQUAL(o2::mcgenstatus::getGenStatusCode(xic.GetStatusCode()), 85);

  int nXiStar = 0;
  int iXiStar = -1;
  int iPi = -1;
  for (int i = 1; i < (int)particles.size(); ++i) {
    if (particles[i].GetPdgCode() == 3324) {
      ++nXiStar;
      iXiStar = i;
    }
    if (particles[i].GetPdgCode() == 211 && particles[i].GetFirstMother() == 0) {
      iPi = i;
    }
  }
  BOOST_REQUIRE_EQUAL(nXiStar, 1);
  BOOST_REQUIRE(iPi > 0);

  // the daughter range of every decayed particle holds exactly its direct products
  for (int m = 0; m < (int)particles.size(); ++m) {
    auto const& mother = particles[m];
    int nChildren = 0;
    for (auto const& p : particles) {
      nChildren += p.GetFirstMother() == m;
    }
    if (hepmc(mother) != 2) {
      BOOST_CHECK_EQUAL(nChildren, 0);
      continue;
    }
    BOOST_CHECK_EQUAL(mother.GetLastDaughter() - mother.GetFirstDaughter() + 1, nChildren);
    for (int d = mother.GetFirstDaughter(); d <= mother.GetLastDaughter(); ++d) {
      BOOST_CHECK_EQUAL(particles[d].GetFirstMother(), m);
    }
  }
  BOOST_CHECK_EQUAL(xic.GetLastDaughter() - xic.GetFirstDaughter() + 1, 2);

  // products carry the Pythia decay status 91 and are flagged as primaries
  for (int i = 1; i < (int)particles.size(); ++i) {
    BOOST_CHECK_EQUAL(o2::mcgenstatus::getGenStatusCode(particles[i].GetStatusCode()), 91);
    BOOST_CHECK_EQUAL(particles[i].GetUniqueID(), kPPrimary);
  }
  auto const& xiStar = particles[iXiStar];
  BOOST_CHECK_EQUAL(xiStar.GetFirstMother(), 0);
  BOOST_CHECK(!toBeDone(xiStar));
  BOOST_CHECK_EQUAL(hepmc(xiStar), 2);

  std::vector<int> dPdg;
  for (int d = xiStar.GetFirstDaughter(); d >= 0 && d <= xiStar.GetLastDaughter(); ++d) {
    dPdg.push_back(particles[d].GetPdgCode());
    BOOST_CHECK(toBeDone(particles[d]));
    BOOST_CHECK_EQUAL(hepmc(particles[d]), 1);
    BOOST_CHECK_EQUAL(particles[d].GetFirstMother(), iXiStar);
  }
  BOOST_REQUIRE_EQUAL(dPdg.size(), 2);
  bool charged = (dPdg[0] == 3312 && dPdg[1] == 211) || (dPdg[0] == 211 && dPdg[1] == 3312);
  bool neutral = (dPdg[0] == 3322 && dPdg[1] == 111) || (dPdg[0] == 111 && dPdg[1] == 3322);
  BOOST_CHECK(charged || neutral);

  int first = xic.GetFirstDaughter();
  int last = xic.GetLastDaughter();
  BOOST_CHECK(first <= iXiStar && iXiStar <= last);
  BOOST_CHECK(first <= iPi && iPi <= last);

  // all products start at the Xic decay point
  double dx = xiStar.Vx() - 0.1;
  double dy = xiStar.Vy() - 0.2;
  double dz = xiStar.Vz() - 0.3;
  BOOST_CHECK(std::sqrt(dx * dx + dy * dy + dz * dz) > 0.);
  for (int i = 1; i < (int)particles.size(); ++i) {
    BOOST_CHECK_CLOSE(particles[i].Vx(), xiStar.Vx(), 1e-6);
    BOOST_CHECK_CLOSE(particles[i].Vy(), xiStar.Vy(), 1e-6);
    BOOST_CHECK_CLOSE(particles[i].Vz(), xiStar.Vz(), 1e-6);
  }

  // products start at the Xic time plus its flight time, |d| / (beta c) with c in cm/ns and T in s
  double flightTime = std::sqrt(dx * dx + dy * dy + dz * dz) / (xic.P() / xic.Energy() * 29.9792458) * 1e-9;
  for (int i = 1; i < (int)particles.size(); ++i) {
    BOOST_CHECK_CLOSE(particles[i].T(), xic.T() + flightTime, 1e-6);
  }
  BOOST_CHECK_EQUAL(report.decayed[4232], 1);
}

BOOST_AUTO_TEST_CASE(antiparticle_chain)
{
  setParams();
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  std::vector<TParticle> particles{makeParticle(-4232, 5., true)};
  auto report = decayer.decay(particles);
  int nAntiXiStar = 0;
  for (auto const& p : particles) {
    nAntiXiStar += p.GetPdgCode() == -3324;
  }
  BOOST_CHECK_EQUAL(nAntiXiStar, 1);
  BOOST_CHECK_EQUAL(report.decayed[-4232], 1);
}

BOOST_AUTO_TEST_CASE(proper_decay_length)
{
  auto const& particles = manyXic();
  double sum = 0.;
  int n = 0;
  for (int i = 0; i < 20000; ++i) {
    auto const& mother = particles[i];
    auto const& daughter = particles[mother.GetFirstDaughter()];
    double dx = daughter.Vx() - mother.Vx();
    double dy = daughter.Vy() - mother.Vy();
    double dz = daughter.Vz() - mother.Vz();
    sum += std::sqrt(dx * dx + dy * dy + dz * dz) * mother.GetCalcMass() / mother.P();
    ++n;
  }
  BOOST_CHECK_CLOSE(sum / n, 0.01358, 2.);
}

BOOST_AUTO_TEST_CASE(xistar_charge_split)
{
  auto const& particles = manyXic();
  int nXiStar = 0;
  int nCharged = 0;
  for (auto const& p : particles) {
    if (p.GetPdgCode() != 3324) {
      continue;
    }
    ++nXiStar;
    for (int d = p.GetFirstDaughter(); d >= 0 && d <= p.GetLastDaughter(); ++d) {
      if (particles[d].GetPdgCode() == 3312) {
        ++nCharged;
      }
    }
  }
  BOOST_REQUIRE_EQUAL(nXiStar, 20000);
  BOOST_CHECK_SMALL(double(nCharged) / nXiStar - 0.667, 0.02);
}

BOOST_AUTO_TEST_CASE(strange_baryons_stay)
{
  setParams();
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  std::vector<TParticle> particles;
  for (int pdg : {3312, 3122, 310, 3334}) {
    particles.push_back(makeParticle(pdg, 2., true));
  }
  auto before = particles;
  auto report = decayer.decay(particles);
  checkUnchanged(particles, before);
  BOOST_CHECK(report.decayed.empty());
}

BOOST_AUTO_TEST_CASE(nothing_to_decay)
{
  setParams();
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  std::vector<TParticle> particles;
  for (int pdg : {211, -321, 2212, 22}) {
    particles.push_back(makeParticle(pdg, 1., true, 0.01, 0.02, 0.03));
  }
  auto before = particles;
  decayer.decay(particles);
  checkUnchanged(particles, before);
}

BOOST_AUTO_TEST_CASE(not_tobedone_untouched)
{
  setParams();
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  std::vector<TParticle> particles{makeParticle(4232, 5., false)};
  auto before = particles;
  auto report = decayer.decay(particles);
  checkUnchanged(particles, before);
  BOOST_CHECK(report.decayed.empty());
}

BOOST_AUTO_TEST_CASE(second_pass_is_noop)
{
  EventDecayerReport report;
  auto particles = decayOneXic(report);
  auto before = particles;
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  auto second = decayer.decay(particles);
  checkUnchanged(particles, before);
  BOOST_CHECK(second.decayed.empty());
}

BOOST_AUTO_TEST_CASE(weight_propagates)
{
  setParams();
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  std::vector<TParticle> particles{makeParticle(4232, 5., true)};
  particles[0].SetWeight(2.5);
  decayer.decay(particles);
  BOOST_REQUIRE(particles.size() > 1);
  for (size_t i = 1; i < particles.size(); ++i) {
    BOOST_CHECK_EQUAL(particles[i].GetWeight(), 2.5);
  }
}

BOOST_AUTO_TEST_CASE(ions_and_unknown_to_pythia)
{
  setParams();
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  std::vector<TParticle> particles{makeParticle(1000020040, 5., true), makeParticle(1010010030, 5., true)};
  auto before = particles;
  auto report = decayer.decay(particles);
  checkUnchanged(particles, before);
  BOOST_CHECK(report.failed.find(1000020040) == report.failed.end());
  BOOST_CHECK(report.failed.find(1010010030) == report.failed.end());
}

BOOST_AUTO_TEST_CASE(impossible_decay)
{
  setParams();
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  auto xic = makeParticle(4232, 5., true);
  // a mass of 1 GeV, below the 3324 + 211 threshold
  xic.SetMomentum(0., 0., 5., std::sqrt(25. + 1.0));
  xic.SetCalcMass(1.0);
  std::vector<TParticle> particles{xic};
  auto before = particles;
  auto report = decayer.decay(particles);
  checkUnchanged(particles, before);
  BOOST_CHECK_EQUAL(report.failed[4232], 1);
  BOOST_CHECK(report.decayed.find(4232) == report.decayed.end());
}

BOOST_AUTO_TEST_CASE(vertex_limits_refused)
{
  for (auto limit : {"limitTau", "limitRadius", "limitCylinder"}) {
    setParams("", writeCfg(std::string("ParticleDecays:") + limit + " = on\n"));
    EventDecayer decayer;
    BOOST_CHECK(decayer.init());
    BOOST_CHECK(!decayer.vertexLimitsActive());
  }
  // the species-level cut of a config is kept when the tau0Max parameter is off
  setParams("", writeCfg("ParticleDecays:limitTau0 = on\nParticleDecays:tau0Max = 0.05\n"));
  o2::conf::ConfigurableParam::setValue("EventDecayer.tau0Max", "0");
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  BOOST_CHECK(!decayer.vertexLimitsActive());
  std::vector<TParticle> particles{makeParticle(4232, 5., true)};
  auto report = decayer.decay(particles);
  BOOST_CHECK_EQUAL(particles.size(), 1);
  BOOST_CHECK(report.decayed.empty());
  BOOST_CHECK(report.failed.empty());
  setParams();
}

BOOST_AUTO_TEST_CASE(pdg_check)
{
  auto table = writeCfg("# test table\n111 pi0 0 0 0\n211 pi+ 7804.5 0 1\n3312 xi- 49.1 0 1\n3322 xi0 87.1 0 1\n");
  setParams(table);
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  std::vector<TParticle> particles{makeParticle(4232, 5., true)};
  auto report = decayer.decay(particles);
  BOOST_CHECK_EQUAL(report.decayed[4232], 1);
  BOOST_CHECK(report.unknownInG4.empty());

  auto unknown = makeParticle(211, 1., true);
  unknown.SetPdgCode(9900001);
  std::vector<TParticle> more{unknown};
  report = decayer.decay(more);
  BOOST_CHECK_EQUAL(report.unknownInO2[9900001], 1);
  BOOST_CHECK_EQUAL(report.unknownInG4[9900001], 1);
}

BOOST_AUTO_TEST_CASE(lifetime_mismatch)
{
  auto table = writeCfg("4232 xi_c+ 0.1300 0 1\n");
  setParams(table);
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  std::vector<TParticle> particles{makeParticle(4232, 5., true)};
  auto report = decayer.decay(particles);
  BOOST_REQUIRE(report.lifetimeMismatch.count(4232) == 1);
  BOOST_CHECK_CLOSE(report.lifetimeMismatch[4232].first, 0.1358, 1e-6);
  BOOST_CHECK_CLOSE(report.lifetimeMismatch[4232].second, 0.1300, 1e-6);
}

BOOST_AUTO_TEST_CASE(units_mm_and_mmc)
{
  // the same decay in cm and s, and in mm and mm/c (the Generator default units)
  const double mmc = 3.3356410e-12;
  std::vector<TParticle> inCm;
  std::vector<TParticle> inMm;
  for (bool mm : {false, true}) {
    setParams("", "", 4711);
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    auto xic = makeParticle(4232, 5., true);
    if (mm) {
      decayer.setUnits(0.1, mmc);
      xic.SetProductionVertex(1., 2., 3., 1e-9 / mmc);
    } else {
      xic.SetProductionVertex(0.1, 0.2, 0.3, 1e-9);
    }
    auto& particles = mm ? inMm : inCm;
    particles.push_back(xic);
    decayer.decay(particles);
  }
  setParams();
  BOOST_REQUIRE(inCm.size() > 1);
  BOOST_REQUIRE_EQUAL(inCm.size(), inMm.size());
  for (size_t i = 1; i < inCm.size(); ++i) {
    BOOST_CHECK_CLOSE(inMm[i].Vz(), 10. * inCm[i].Vz(), 1e-6);
    BOOST_CHECK_CLOSE(inMm[i].T(), inCm[i].T() / mmc, 1e-6);
  }
  // in mm and mm/c the flight time is the decay length over beta
  auto const& xic = inMm[0];
  auto const& d = inMm[xic.GetFirstDaughter()];
  double length = std::sqrt(std::pow(d.Vx() - 1., 2) + std::pow(d.Vy() - 2., 2) + std::pow(d.Vz() - 3., 2));
  BOOST_CHECK(length > 0.);
  BOOST_CHECK_CLOSE(d.T() - xic.T(), length / (xic.P() / xic.Energy()), 1e-4);
}

BOOST_AUTO_TEST_CASE(seeding)
{
  // decays one Xic with the given seed and returns the seed in use and the decay position
  auto run = [](int seed) {
    setParams("", "", seed);
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    std::vector<TParticle> particles{makeParticle(4232, 5., true)};
    decayer.decay(particles);
    BOOST_REQUIRE(particles.size() > 1);
    return std::make_pair(decayer.seed(), particles[1].Vz());
  };
  // a seed above the Pythia8 maximum is taken modulo 900000001
  auto large = run(1000000123);
  auto reduced = run(100000122);
  BOOST_CHECK_EQUAL(large.first, 100000122);
  BOOST_CHECK_EQUAL(large.second, reduced.second);
  // seed 0 derives the seed from gRandom, differently from GeneratorPythia8 which takes it unchanged
  gRandom->SetSeed(777);
  auto fromGRandom = run(0);
  BOOST_CHECK(fromGRandom.first != 777);
  BOOST_CHECK(fromGRandom.first >= 0 && fromGRandom.first <= 900000000);
  auto explicitSeed = run(fromGRandom.first);
  BOOST_CHECK_EQUAL(fromGRandom.second, explicitSeed.second);
  gRandom->SetSeed(777);
  BOOST_CHECK_EQUAL(run(0).first, fromGRandom.first);
  gRandom->SetSeed(778);
  BOOST_CHECK(run(0).first != fromGRandom.first);
  BOOST_CHECK(run(1).second != run(2).second);
  setParams();
}

// quarks, gluon, diquarks and the Pythia8 pseudo-particles 81-99
bool isPartonic(int pdg)
{
  int a = std::abs(pdg);
  bool diquark = a > 1100 && a < 5600 && a / 1000 < 6 && (a / 10) % 10 == 0 && (a / 100) % 10 <= a / 1000 && (a / 100) % 10 > 0 && (a % 10) % 2 == 1;
  return (a >= 1 && a <= 8) || a == 21 || (a >= 81 && a <= 99) || diquark;
}

BOOST_AUTO_TEST_CASE(partonic_decay_never_transports_partons)
{
  // Xic+ -> s uu_1 with meMode 91; whether Pythia8 hadronises the partons depends on its version (8.318 does, 8.315 does not)
  setParams("", writeCfg("4232:oneChannel = 1 1.0 91 3 2203\n"));
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  std::vector<TParticle> particles{makeParticle(4232, 5., true)};
  auto before = particles;
  auto report = decayer.decay(particles);
  for (auto const& p : particles) {
    BOOST_CHECK(!(toBeDone(p) && isPartonic(p.GetPdgCode())));
  }
  if (report.failed[4232] == 1) {
    checkUnchanged(particles, before);
    BOOST_CHECK(report.decayed.empty());
  } else {
    BOOST_CHECK_EQUAL(report.decayed[4232], 1);
    BOOST_CHECK(!toBeDone(particles[0]));
    BOOST_CHECK(particles.size() > 1);
  }
}

BOOST_AUTO_TEST_CASE(failed_intermediate_decay_fails)
{
  // Xi(1530)0 -> Xi- p is closed, so Pythia8 cannot decay the intermediate
  setParams("", writeCfg("3324:oneChannel = 1 1.0 0 3312 2212\n"));
  EventDecayer decayer;
  BOOST_REQUIRE(decayer.init());
  std::vector<TParticle> particles{makeParticle(4232, 5., true)};
  auto before = particles;
  auto report = decayer.decay(particles);
  checkUnchanged(particles, before);
  BOOST_CHECK_EQUAL(report.failed[4232], 1);
  BOOST_CHECK(report.decayed.empty());
}

BOOST_AUTO_TEST_CASE(config_fallback_to_decayerpythia8)
{
  static const std::string forcePath = writeCfg(FORCE);
  auto clear = []() {
    for (int i = 0; i < 8; ++i) {
      o2::conf::ConfigurableParam::setValue("EventDecayer.config[" + std::to_string(i) + "]", "");
    }
  };
  // with every EventDecayer.config empty the DecayerPythia8 configs are read
  setParams();
  clear();
  o2::conf::ConfigurableParam::setValue("DecayerPythia8.config[1]", forcePath);
  {
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    auto summary = decayer.configSummary();
    BOOST_CHECK(summary.find("pythia8/decays/base.cfg;" + forcePath) != std::string::npos);
    std::vector<TParticle> particles{makeParticle(4232, 5., true)};
    BOOST_CHECK_EQUAL(decayer.decay(particles).decayed[4232], 1);
  }
  // an explicitly set EventDecayer.config wins over the DecayerPythia8 one of the same slot
  auto other = writeCfg("4232:mayDecay = off\n");
  o2::conf::ConfigurableParam::setValue("EventDecayer.config[1]", other);
  {
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    auto summary = decayer.configSummary();
    BOOST_CHECK(summary.find("pythia8/decays/base.cfg;" + other) != std::string::npos);
    BOOST_CHECK(summary.find(forcePath) == std::string::npos);
    std::vector<TParticle> particles{makeParticle(4232, 5., true)};
    BOOST_CHECK(decayer.decay(particles).decayed.empty());
  }
  o2::conf::ConfigurableParam::setValue("DecayerPythia8.config[1]", "");
  setParams();
}

BOOST_AUTO_TEST_CASE(tau0max_leaves_long_lived)
{
  // Xic+ (tau0 0.1358 mm) above tau0Max stays, and is not counted as failed
  setParams();
  o2::conf::ConfigurableParam::setValue("EventDecayer.tau0Max", "0.1");
  {
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    std::vector<TParticle> particles{makeParticle(4232, 5., true)};
    auto before = particles;
    auto report = decayer.decay(particles);
    checkUnchanged(particles, before);
    BOOST_CHECK(report.decayed.empty());
    BOOST_CHECK(report.failed.empty());
  }
  // a decayable product above tau0Max (Xi-, Xi0) is left for transport
  setParams("", writeCfg("3312:mayDecay = on\n3322:mayDecay = on\n"));
  {
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    std::vector<TParticle> particles{makeParticle(4232, 5., true)};
    auto report = decayer.decay(particles);
    BOOST_CHECK_EQUAL(report.decayed[4232], 1);
    int nXi = 0;
    for (auto const& p : particles) {
      if (isXi(p)) {
        ++nXi;
        BOOST_CHECK(toBeDone(p));
        BOOST_CHECK_EQUAL(hepmc(p), 1);
      }
    }
    BOOST_CHECK_EQUAL(nXi, 1);
  }
  // tau0Max = 0 switches the cut off, so the Xi decays as well
  o2::conf::ConfigurableParam::setValue("EventDecayer.tau0Max", "0");
  {
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    std::vector<TParticle> particles{makeParticle(4232, 5., true)};
    decayer.decay(particles);
    for (auto const& p : particles) {
      if (isXi(p)) {
        BOOST_CHECK_EQUAL(hepmc(p), 2);
      }
    }
  }
  setParams();
}

BOOST_AUTO_TEST_CASE(skip_pdg)
{
  // a listed code is never decayed, for the particle and its antiparticle
  setParams();
  o2::conf::ConfigurableParam::setValue("EventDecayer.skipPdg", "421  4232");
  {
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    std::vector<TParticle> particles{makeParticle(4232, 5., true), makeParticle(-4232, 5., true)};
    auto before = particles;
    auto report = decayer.decay(particles);
    checkUnchanged(particles, before);
    BOOST_CHECK(report.decayed.empty());
    BOOST_CHECK(report.failed.empty());
  }
  // a listed product stays undecayed and is left for transport
  o2::conf::ConfigurableParam::setValue("EventDecayer.skipPdg", "3324");
  {
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    std::vector<TParticle> particles{makeParticle(4232, 5., true)};
    auto report = decayer.decay(particles);
    BOOST_CHECK_EQUAL(report.decayed[4232], 1);
    BOOST_REQUIRE_EQUAL(particles.size(), 3);
    for (size_t i = 1; i < particles.size(); ++i) {
      BOOST_CHECK(toBeDone(particles[i]));
      BOOST_CHECK_EQUAL(hepmc(particles[i]), 1);
    }
  }
  // the default list keeps the Lambda(1520) for its Geant4 decay modes in O2
  setParams();
  {
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    std::vector<TParticle> particles{makeParticle(102134, 2., true), makeParticle(-102134, 2., true)};
    auto report = decayer.decay(particles);
    BOOST_CHECK_EQUAL(particles.size(), 2);
    BOOST_CHECK(report.decayed.empty());
  }
  // with an empty list Pythia8 decays it
  o2::conf::ConfigurableParam::setValue("EventDecayer.skipPdg", "");
  {
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    std::vector<TParticle> particles{makeParticle(102134, 2., true)};
    auto report = decayer.decay(particles);
    BOOST_CHECK_EQUAL(report.decayed[102134], 1);
  }
  setParams();
}

BOOST_AUTO_TEST_CASE(units_momentum_and_energy)
{
  // the same decay in GeV and in MeV
  std::vector<TParticle> inGeV;
  std::vector<TParticle> inMeV;
  for (bool mev : {false, true}) {
    setParams("", "", 4711);
    EventDecayer decayer;
    BOOST_REQUIRE(decayer.init());
    auto xic = makeParticle(4232, 5., true);
    if (mev) {
      decayer.setUnits(1., 1., 1e-3, 1e-3);
      xic.SetMomentum(1e3 * xic.Px(), 1e3 * xic.Py(), 1e3 * xic.Pz(), 1e3 * xic.Energy());
    }
    auto& particles = mev ? inMeV : inGeV;
    particles.push_back(xic);
    decayer.decay(particles);
  }
  setParams();
  BOOST_REQUIRE(inGeV.size() > 1);
  BOOST_REQUIRE_EQUAL(inGeV.size(), inMeV.size());
  for (size_t i = 1; i < inGeV.size(); ++i) {
    BOOST_CHECK_EQUAL(inMeV[i].GetPdgCode(), inGeV[i].GetPdgCode());
    BOOST_CHECK_CLOSE(inMeV[i].Pz(), 1e3 * inGeV[i].Pz(), 1e-6);
    BOOST_CHECK_CLOSE(inMeV[i].Energy(), 1e3 * inGeV[i].Energy(), 1e-6);
    BOOST_CHECK_CLOSE(inMeV[i].Vz(), inGeV[i].Vz(), 1e-6);
  }
}
