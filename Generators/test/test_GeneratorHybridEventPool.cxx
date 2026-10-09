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

/// \file test_GeneratorHybridEventPool.cxx
/// \brief regression tests for the use of event pools (and finite inputs in general) inside the GeneratorHybrid
/// \author M. Giacalone, mgiacalo@cern.ch, 10/2026
///
/// The GeneratorHybrid is a process-wide singleton and the tests of the failure modes end the process, so
/// every scenario runs in a child process: this very test executable, started again with the scenario in the
/// environment (see HybridScenarioWorker at the end).

#define BOOST_TEST_MODULE Test GeneratorHybridEventPool
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "EventPoolTestUtils.h"
#include <CommonUtils/ConfigurableParam.h>
#include <Generators/GeneratorHybrid.h>
#include <TRandom.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/O2DatabasePDG.h>

#include <TDatabasePDG.h>
#include <TParticle.h>

#include <HepMC3/GenEvent.h>
#include <HepMC3/GenParticle.h>
#include <HepMC3/GenVertex.h>
#include <HepMC3/WriterAscii.h>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace evtpooltest;
using o2::eventgen::Generator;
using o2::eventgen::GeneratorFromEventPool;
using o2::eventgen::GeneratorHybrid;

namespace
{

constexpr const char* kScenarioEnv = "O2_HYBRIDTEST_SCENARIO"; // scenario to run in the child process
constexpr const char* kDirEnv = "O2_HYBRIDTEST_DIR";           // scratch directory of the child process
constexpr const char* kWindowEnv = "O2_HYBRIDTEST_WINDOW_MS";  // stretching of the race window (0 = natural timing)
constexpr int kFatalExitCode = 86;                             // exit code of the child on LOG(fatal) (too_few_events, uneven_shares)
constexpr int kWatchdogExitCode = 87;                          // exit code of a child which did not end in time (e.g. a hang)
constexpr int kWatchdogSeconds = 180;
constexpr int kDefaultWindowMs = 300;

// ---------------------------------------------------------------------------------------------------------------
// building of the configuration of the hybrid generator
// ---------------------------------------------------------------------------------------------------------------

std::string boxJson(int pdg, int number)
{
  // the hybrid reads the configuration with TBufferJSON, which needs every member of BoxGenConfig
  std::ostringstream s;
  s << R"({ "name": "boxgen", "config": { "pdg": )" << pdg << R"(, "number": )" << number
    << R"(, "eta": [-1, 1], "prange": [0.1, 5], "phirange": [0, 360], "sampleYAndPt": false } })";
  return s.str();
}

std::string poolJson(std::string const& path)
{
  std::ostringstream s;
  s << R"({ "name": "evtpool", "config": { "eventPoolPath": ")" << path
    << R"(", "skipNonTrackable": true, "roundRobin": false, "randomize": false, "rngseed": 1, "randomphi": false } })";
  return s.str();
}

/// a HepMC sub-generator reading one file; with randomize it serves every event of the file once, in random order
std::string hepmcJson(std::string const& file, bool randomize)
{
  // as for the box generator, every member of FileOrCmdGenConfig and HepMCGenConfig must be given
  std::ostringstream s;
  s << R"({ "name": "hepmc", "config": { "configcmd": { "fileNames": ")" << file << R"(", "cmd": "" }, )"
    << R"("confighepmc": { "version": 0, "eventsToSkip": 0, "fileName": "", "prune": false, "randomize": )"
    << (randomize ? "true" : "false")
    << R"(, "roundRobin": false, "reshuffleOnRepeat": false, "rngseed": 1 } } })";
  return s.str();
}

/// writes a HepMC3 ASCII file with nevents events; event i holds a proton beam particle and one K+ with px = i + 1
void createHepMCFile(std::string const& path, int nevents)
{
  HepMC3::WriterAscii writer(path);
  for (int i = 0; i < nevents; ++i) {
    HepMC3::GenEvent event(HepMC3::Units::GEV, HepMC3::Units::MM);
    event.set_event_number(i);
    const double px = i + 1.;
    const double mK = 0.493677;
    auto beam = std::make_shared<HepMC3::GenParticle>(HepMC3::FourVector(0., 0., 100., std::sqrt(100. * 100. + 0.938272 * 0.938272)), 2212, 4);
    auto kaon = std::make_shared<HepMC3::GenParticle>(HepMC3::FourVector(px, 0., 0., std::sqrt(px * px + mK * mK)), 321, 1);
    auto vertex = std::make_shared<HepMC3::GenVertex>();
    vertex->add_particle_in(beam);
    vertex->add_particle_out(kaon);
    event.add_vertex(vertex);
    writer.write_event(event);
  }
  writer.close();
}

std::string cocktailJson(std::vector<std::string> const& subs)
{
  std::string s = R"({ "cocktail": [ )";
  for (size_t i = 0; i < subs.size(); ++i) {
    s += (i ? ", " : "") + subs[i];
  }
  return s + " ] }";
}

std::string hybridJson(std::vector<std::string> const& entries, std::string const& fractions, std::string const& mode = "")
{
  std::string s = "{\n";
  if (!mode.empty()) {
    s += "  \"mode\": \"" + mode + "\",\n";
  }
  s += "  \"generators\": [\n";
  for (size_t i = 0; i < entries.size(); ++i) {
    s += "    " + entries[i] + (i + 1 < entries.size() ? ",\n" : "\n");
  }
  s += "  ],\n  \"fractions\": [ " + fractions + " ]\n}\n";
  return s;
}

std::string writeFile(fs::path const& path, std::string const& content)
{
  fs::create_directories(path.parent_path());
  std::ofstream out(path);
  out << content;
  return path.string();
}

// ---------------------------------------------------------------------------------------------------------------
// helpers of the child process
// ---------------------------------------------------------------------------------------------------------------

/// Stretches the time between the scheduling of the lookahead task of the last event and the raising of the stop
/// flag (see the file description). Does nothing if `ms` is 0.
class RaceWindowWidener
{
 public:
  explicit RaceWindowWidener(int ms) : mMs(ms)
  {
    if (mMs > 0) {
      fair::Logger::AddCustomSink("race_window_widener", fair::Severity::info, [this](std::string const& content, fair::LogMetaData const&) {
        if (content.find("HybridGen: Stopping TBB task pool") != std::string::npos) {
          mTriggered = true;
          std::this_thread::sleep_for(std::chrono::milliseconds(mMs));
        }
      });
    }
  }
  ~RaceWindowWidener()
  {
    if (mMs > 0) {
      fair::Logger::RemoveCustomSink("race_window_widener");
    }
  }
  /// whether the stretched window was entered; if not (e.g. because the message above was reworded) the
  /// scenarios would pass without actually testing anything
  bool triggered() const { return mTriggered; }
  bool enabled() const { return mMs > 0; }

 private:
  int mMs;
  std::atomic<bool> mTriggered{false};
};

void configureHybrid(int numWorkers, bool randomize)
{
  o2::eventgen::GeneratorHybridParam::Instance(); // makes sure that the parameters are registered
  o2::conf::ConfigurableParam::updateFromString("GeneratorHybrid.num_workers=" + std::to_string(numWorkers) +
                                                ";GeneratorHybrid.randomize=" + (randomize ? "true" : "false"));
}

/// what the job gets for one event
struct EventRecord {
  std::vector<TParticle> particles;
  std::string forwardingGenerator; // as written to the event header
};

/// runs the event loop of the job: the two calls are what the primary generator does for every event
std::vector<EventRecord> produce(GeneratorHybrid& hybrid, unsigned int nevents)
{
  std::vector<EventRecord> events;
  for (unsigned int i = 0; i < nevents; ++i) {
    BOOST_REQUIRE(hybrid.generateEvent());
    BOOST_REQUIRE(hybrid.importParticles());
    EventRecord record;
    record.particles = hybrid.getParticles();
    o2::dataformats::MCEventHeader header;
    hybrid.updateHeader(&header);
    bool valid = false;
    record.forwardingGenerator = header.getInfo<std::string>("forwarding-generator", valid);
    events.push_back(std::move(record));
  }
  return events;
}

/// gives the worker threads the time to do whatever they still want to do after the job has ended
void settle() { std::this_thread::sleep_for(std::chrono::milliseconds(300)); }

/// identifier of the pool event (first track of the pool event; see encodeMomentum), -1 if the event has none
double poolEventId(EventRecord const& event)
{
  for (auto const& p : event.particles) {
    if (p.GetPdgCode() == 211) {
      return p.Px();
    }
  }
  return -1.;
}

/// identifiers of all the events of a pool made with createPool(…, nfiles)
std::set<double> allPoolEventIds(int nfiles)
{
  std::set<double> ids;
  for (int f = 0; f < nfiles; ++f) {
    for (int e = 0; e < eventsInFile(f); ++e) {
      ids.insert(encodeMomentum(f, e));
    }
  }
  return ids;
}

int poolEventsInFiles(int nfiles)
{
  int n = 0;
  for (int f = 0; f < nfiles; ++f) {
    n += eventsInFile(f);
  }
  return n;
}

GeneratorFromEventPool* subPool(GeneratorHybrid& hybrid, size_t index)
{
  auto const& gens = hybrid.getGenerators();
  BOOST_REQUIRE(index < gens.size());
  auto pool = dynamic_cast<GeneratorFromEventPool*>(gens[index].get());
  BOOST_REQUIRE(pool != nullptr);
  return pool;
}

struct ScenarioEnv {
  fs::path dir;
  int windowMs = kDefaultWindowMs;
};

/// the number of events each sub-generator of the hybrid was told to serve
std::vector<unsigned int> expectedShares(GeneratorHybrid& hybrid)
{
  std::vector<unsigned int> shares;
  for (auto const& gen : hybrid.getGenerators()) {
    shares.push_back(gen->getExpectedNEvents());
  }
  return shares;
}

/// common start of the scenarios: configures the generator and creates the singleton
GeneratorHybrid& startHybrid(ScenarioEnv const& env, std::string const& json, unsigned int nevents, int numWorkers = 2, bool randomize = false)
{
  auto jsonPath = writeFile(env.dir / "hybrid.json", json);
  // the (lazily built, not thread-safe) particle database must be complete before several worker threads start to
  // use it; Instance() also marks it as initialised, so that the generators calling O2DatabasePDG::Instance() in the
  // worker threads do not add the ALICE particles again
  o2::O2DatabasePDG::Instance();
  configureHybrid(numWorkers, randomize);
  static unsigned int total; // setTotalNEvents wants an lvalue
  total = nevents;
  Generator::setTotalNEvents(total);
  return GeneratorHybrid::Instance(jsonPath);
}

// ---------------------------------------------------------------------------------------------------------------
// the scenarios; each one runs in a process of its own
// ---------------------------------------------------------------------------------------------------------------

/// The job asks for exactly as many events as the pool holds (9 events in 3 files). Every one of them must be
/// served exactly once, and the pool must not be asked for another one after the last event of the job.
void scenarioExactPool(ScenarioEnv const& env)
{
  LogCatcher log("scenario_exact_pool"); // warnings and worse; LOG(fatal) throws
  RaceWindowWidener widener(env.windowMs);
  constexpr int nfiles = 3;
  auto poolDir = env.dir / "pool";
  createPool(poolDir, nfiles);
  const unsigned int nevents = poolEventsInFiles(nfiles);
  auto& hybrid = startHybrid(env, hybridJson({poolJson(poolDir.string())}, "1"), nevents);
  BOOST_REQUIRE(hybrid.Init());
  // the only generator serves all the events, and the pool passes this on to the generator reading the files
  BOOST_CHECK((expectedShares(hybrid) == std::vector<unsigned int>{nevents}));
  BOOST_CHECK_EQUAL(subPool(hybrid, 0)->getO2KineGenerator()->getExpectedNEvents(), nevents);
  BOOST_CHECK_EQUAL(log.count("holds only about"), 0);

  auto events = produce(hybrid, nevents);
  settle();

  BOOST_REQUIRE_EQUAL(events.size(), nevents);
  std::set<double> ids;
  for (auto const& e : events) {
    ids.insert(poolEventId(e));
    BOOST_CHECK_EQUAL(e.forwardingGenerator, "HybridGen");
  }
  BOOST_CHECK(ids == allPoolEventIds(nfiles)); // every event of the pool exactly once
  auto pool = subPool(hybrid, 0);
  std::cout << "HYBRIDTEST served=" << pool->getO2KineGenerator()->getEventsServed() << " requested=" << nevents << std::endl;
  BOOST_CHECK_EQUAL(pool->getO2KineGenerator()->getEventsServed(), static_cast<int>(nevents));
  BOOST_CHECK_EQUAL(pool->getO2KineGenerator()->getNumberOfFilesUsed(), nfiles);
  BOOST_CHECK_EQUAL(log.count("ran out of events"), 0);
  BOOST_CHECK_MESSAGE(!widener.enabled() || widener.triggered(), "the race window was not entered: the scenario tests nothing");
}

/// The pool holds more events than the job asks for (14 against 9). The pool must have delivered exactly the
/// 9 events which were consumed: with the lookahead after the last event it delivers 10. This is the same
/// check as above, but it does not depend on the fatal error of a pool which is used up. With randomize the
/// shares are not exact, there is no request budget, and only the guard on the last event of the job helps.
void scenarioLargerPool(ScenarioEnv const& env, bool randomize)
{
  LogCatcher log("scenario_larger_pool");
  RaceWindowWidener widener(env.windowMs);
  constexpr int nfiles = 4;
  auto poolDir = env.dir / "pool";
  createPool(poolDir, nfiles);
  const unsigned int nevents = 9;
  auto& hybrid = startHybrid(env, hybridJson({poolJson(poolDir.string())}, "1"), nevents, 2, randomize);
  BOOST_REQUIRE(hybrid.Init());

  auto events = produce(hybrid, nevents);
  settle();

  BOOST_REQUIRE_EQUAL(events.size(), nevents);
  std::set<double> ids;
  for (auto const& e : events) {
    ids.insert(poolEventId(e));
  }
  BOOST_CHECK_EQUAL(ids.size(), static_cast<size_t>(nevents)); // no event twice
  auto pool = subPool(hybrid, 0);
  std::cout << "HYBRIDTEST served=" << pool->getO2KineGenerator()->getEventsServed() << " requested=" << nevents << std::endl;
  BOOST_CHECK_EQUAL(pool->getO2KineGenerator()->getEventsServed(), static_cast<int>(nevents));
  BOOST_CHECK_MESSAGE(!widener.enabled() || widener.triggered(), "the race window was not entered: the scenario tests nothing");
}

/// Cocktail: every event merges a box generator (2 muons) and an event pool (a chain of tracks), the pool
/// holds exactly as many events as the job asks for. Checks the merging (order, mother/daughter indices),
/// the single use of the pool events and - as for the plain mode - that nothing is scheduled after the last
/// event of the job, which in the cocktail code is a place of its own.
void scenarioCocktailPool(ScenarioEnv const& env)
{
  LogCatcher log("scenario_cocktail_pool");
  RaceWindowWidener widener(env.windowMs);
  constexpr int nfiles = 3;
  constexpr int nbox = 2;
  auto poolDir = env.dir / "pool";
  createPool(poolDir, nfiles, /*chain=*/true);
  const unsigned int nevents = poolEventsInFiles(nfiles);
  auto json = hybridJson({cocktailJson({boxJson(13, nbox), poolJson(poolDir.string())})}, "1");
  auto& hybrid = startHybrid(env, json, nevents);
  BOOST_REQUIRE(hybrid.Init());
  // every member of the cocktail contributes to every event
  BOOST_CHECK((expectedShares(hybrid) == std::vector<unsigned int>{nevents, nevents}));
  BOOST_CHECK_EQUAL(log.count("holds only about"), 0);

  auto events = produce(hybrid, nevents);
  settle();

  BOOST_REQUIRE_EQUAL(events.size(), nevents);
  std::set<double> ids;
  for (auto const& e : events) {
    auto const& ps = e.particles;
    BOOST_REQUIRE_GE(ps.size(), static_cast<size_t>(nbox + 1));
    // the particles of the first generator of the cocktail come first, then the ones of the pool
    for (int i = 0; i < nbox; ++i) {
      BOOST_CHECK_EQUAL(ps[i].GetPdgCode(), 13);
      BOOST_CHECK_EQUAL(ps[i].GetMother(0), -1);
    }
    const int npool = ps.size() - nbox;
    ids.insert(ps[nbox].Px());
    for (int j = 0; j < npool; ++j) {
      auto const& p = ps[nbox + j];
      BOOST_CHECK_EQUAL(p.GetPdgCode(), 211);
      BOOST_CHECK_CLOSE(p.Px(), ps[nbox].Px(), 1E-6); // all tracks of a pool event carry its identifier
      // the chain of the pool event is shifted by the particles which came before it
      BOOST_CHECK_EQUAL(p.GetMother(0), j > 0 ? nbox + j - 1 : -1);
      BOOST_CHECK_EQUAL(p.GetDaughter(0), j < npool - 1 ? nbox + j + 1 : -1);
    }
    BOOST_CHECK_EQUAL(e.forwardingGenerator, "HybridGen");
  }
  BOOST_CHECK(ids == allPoolEventIds(nfiles));
  auto pool = subPool(hybrid, 1);
  std::cout << "HYBRIDTEST served=" << pool->getO2KineGenerator()->getEventsServed() << " requested=" << nevents << std::endl;
  BOOST_CHECK_EQUAL(pool->getO2KineGenerator()->getEventsServed(), static_cast<int>(nevents));
  BOOST_CHECK_EQUAL(log.count("ran out of events"), 0);
  BOOST_CHECK_MESSAGE(!widener.enabled() || widener.triggered(), "the race window was not entered: the scenario tests nothing");
}

/// Cocktail and plain generators next to each other: group 0 is the cocktail of three box generators (2 muons,
/// 1 pion), group 1 is a single photon; the fractions make the two alternate.
void scenarioCocktailGroups(ScenarioEnv const& env)
{
  LogCatcher log("scenario_cocktail_groups");
  RaceWindowWidener widener(env.windowMs);
  const unsigned int nevents = 6;
  auto json = hybridJson({cocktailJson({boxJson(13, 2), boxJson(211, 1)}), boxJson(22, 1)}, "1, 1");
  auto& hybrid = startHybrid(env, json, nevents);
  BOOST_REQUIRE(hybrid.Init());
  // the fraction is the one of the group, it applies to every member of it
  BOOST_CHECK((expectedShares(hybrid) == std::vector<unsigned int>{3, 3, 3}));

  auto events = produce(hybrid, nevents);
  settle();

  BOOST_REQUIRE_EQUAL(events.size(), nevents);
  for (unsigned int i = 0; i < nevents; ++i) {
    std::vector<int> pdgs;
    for (auto const& p : events[i].particles) {
      pdgs.push_back(p.GetPdgCode());
    }
    std::sort(pdgs.begin(), pdgs.end());
    if (i % 2 == 0) {
      BOOST_CHECK((pdgs == std::vector<int>{13, 13, 211}));
    } else {
      BOOST_CHECK((pdgs == std::vector<int>{22}));
    }
  }
  BOOST_CHECK_MESSAGE(!widener.enabled() || widener.triggered(), "the race window was not entered: the scenario tests nothing");
}

/// Parallel mode: clones of one generator share the work
void scenarioParallelMode(ScenarioEnv const& env)
{
  LogCatcher log("scenario_parallel_mode");
  RaceWindowWidener widener(env.windowMs);
  const unsigned int nevents = 7;
  auto json = hybridJson({boxJson(13, 1), boxJson(13, 1), boxJson(13, 1)}, "1, 1, 1", "parallel");
  auto& hybrid = startHybrid(env, json, nevents, 3);
  BOOST_REQUIRE(hybrid.Init());
  // any clone can serve any event: the share of each one is not known
  BOOST_CHECK((expectedShares(hybrid) == std::vector<unsigned int>{0, 0, 0}));

  auto events = produce(hybrid, nevents);
  settle();

  BOOST_REQUIRE_EQUAL(events.size(), nevents);
  for (auto const& e : events) {
    BOOST_REQUIRE_EQUAL(e.particles.size(), 1u);
    BOOST_CHECK_EQUAL(e.particles[0].GetPdgCode(), 13);
  }
  BOOST_CHECK_MESSAGE(!widener.enabled() || widener.triggered(), "the race window was not entered: the scenario tests nothing");
}

/// An event pool next to a box generator, the job asks for more events (10) than the pool holds (9): the pool
/// only serves its share (5). The pool must be told its share and not the number of events of the whole job,
/// i.e. it must not warn that its input is too small.
void scenarioMixedSequence(ScenarioEnv const& env)
{
  LogCatcher log("scenario_mixed_sequence");
  RaceWindowWidener widener(env.windowMs);
  auto poolFile = (env.dir / "pool" / "evtpool.root").string();
  fs::create_directories(env.dir / "pool");
  createKineFile(poolFile, 9, 0);
  const unsigned int nevents = 10;
  auto& hybrid = startHybrid(env, hybridJson({poolJson(poolFile), boxJson(13, 1)}, "1, 1"), nevents);
  BOOST_REQUIRE(hybrid.Init());

  // fractions 1:1 over 10 events
  BOOST_CHECK((expectedShares(hybrid) == std::vector<unsigned int>{5, 5}));
  BOOST_CHECK_EQUAL(subPool(hybrid, 0)->getO2KineGenerator()->getExpectedNEvents(), 5u);

  auto events = produce(hybrid, nevents);
  settle();

  BOOST_REQUIRE_EQUAL(events.size(), nevents);
  int fromPool = 0;
  std::set<double> ids;
  for (auto const& e : events) {
    if (poolEventId(e) >= 0.) {
      fromPool++;
      ids.insert(poolEventId(e));
    }
  }
  BOOST_CHECK_EQUAL(fromPool, 5);
  BOOST_CHECK_EQUAL(ids.size(), 5u); // the 5 events are different ones
  BOOST_CHECK_EQUAL(log.count("holds only about"), 0);
  BOOST_CHECK_EQUAL(log.count("ran out of events"), 0);
  BOOST_CHECK_MESSAGE(!widener.enabled() || widener.triggered(), "the race window was not entered: the scenario tests nothing");
}

/// Fractions 2:1 over 10 events: the generators are used in turn, 2 events from the pool, then 1 from the
/// box, so the pool serves 7 events (2+2+2 and the first of the last cycle) and the box 3. A pool holding 6
/// events is too small for this share: the pool says so at initialisation, and the job stops with
/// 'ran out of events', reporting the share of the pool. The child ends with kFatalExitCode, the parent checks
/// its output.
void scenarioUnevenShares(ScenarioEnv const& env)
{
  fair::Logger::OnFatal([] {
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(kFatalExitCode);
  });
  auto poolFile = (env.dir / "pool" / "evtpool.root").string();
  fs::create_directories(env.dir / "pool");
  createKineFile(poolFile, 6, 0);
  auto& hybrid = startHybrid(env, hybridJson({poolJson(poolFile), boxJson(13, 1)}, "2, 1"), 10);
  BOOST_REQUIRE(hybrid.Init());
  auto shares = expectedShares(hybrid);
  std::cout << "HYBRIDTEST shares=";
  for (auto n : shares) {
    std::cout << n << ",";
  }
  std::cout << " kine=" << subPool(hybrid, 0)->getO2KineGenerator()->getExpectedNEvents() << std::endl;
  produce(hybrid, 10); // does not come back
  BOOST_ERROR("the job went through although the pool holds fewer events than its share");
}

/// With randomize the generator of each event is drawn with probability fraction/sum: the share is not known
/// in advance, the mean (rounded up) is used. Fractions 3:1 over 10 events: 7.5 and 2.5, i.e. 8 and 3.
void scenarioRandomShares(ScenarioEnv const& env)
{
  LogCatcher log("scenario_random_shares");
  const unsigned int nevents = 10;
  auto& hybrid = startHybrid(env, hybridJson({boxJson(13, 1), boxJson(22, 1)}, "3, 1"), nevents, 2, /*randomize=*/true);
  BOOST_REQUIRE(hybrid.Init());
  BOOST_CHECK((expectedShares(hybrid) == std::vector<unsigned int>{8, 3}));

  // run the job to its end, so that the worker threads stop before the generator is destroyed
  auto events = produce(hybrid, nevents);
  settle();
  BOOST_CHECK_EQUAL(events.size(), nevents);
}

/// An event pool which does not exist stops the job at initialisation, instead of being used uninitialised later
/// on. The pool says so itself (as it does outside a hybrid), before the hybrid checks the result of its Init().
void scenarioInitFailure(ScenarioEnv const& env)
{
  LogCatcher log("scenario_init_failure", fair::Severity::error);
  auto& hybrid = startHybrid(env, hybridJson({poolJson((env.dir / "does_not_exist").string())}, "1"), 5);
  BOOST_CHECK_THROW(hybrid.Init(), fair::FatalException);
  BOOST_CHECK_EQUAL(log.count("No file found that can be used with EventPool generator"), 1);
}

/// A sub-generator whose Init() only reports a failure through its result (here: GeneratorHepMC with an input
/// file which does not exist) stops the job at initialisation as well: the hybrid checks that result.
void scenarioInitFailureHepMC(ScenarioEnv const& env)
{
  LogCatcher log("scenario_init_failure_hepmc", fair::Severity::error);
  auto json = hybridJson({hepmcJson((env.dir / "does_not_exist.hepmc").string(), false), boxJson(13, 1)}, "1, 1");
  auto& hybrid = startHybrid(env, json, 5);
  BOOST_CHECK_THROW(hybrid.Init(), fair::FatalException);
  BOOST_CHECK_EQUAL(log.count("Initialization of sub-generator 0 (hepmc) failed"), 1);
}

/// A HepMC file (5 events, served in random order, each once) next to a box generator, fractions 1:1 over 10
/// events: the HepMC generator only serves 5 events. It must be told this share, and not the 10 events of the
/// job: no "input too small" warning, and, as for the pools, it is not asked for a 6th event.
void scenarioHepMCShare(ScenarioEnv const& env)
{
  LogCatcher log("scenario_hepmc_share");
  auto hepmcFile = (env.dir / "events.hepmc").string();
  fs::create_directories(env.dir);
  createHepMCFile(hepmcFile, 5);
  const unsigned int nevents = 10;
  auto& hybrid = startHybrid(env, hybridJson({hepmcJson(hepmcFile, true), boxJson(13, 1)}, "1, 1"), nevents);
  BOOST_REQUIRE(hybrid.Init());
  BOOST_CHECK((expectedShares(hybrid) == std::vector<unsigned int>{5, 5}));
  BOOST_CHECK_EQUAL(log.count("This job will request"), 0);

  auto events = produce(hybrid, nevents);
  settle();

  BOOST_REQUIRE_EQUAL(events.size(), nevents);
  std::set<double> kaons;
  int fromBox = 0;
  for (auto const& e : events) {
    for (auto const& p : e.particles) {
      if (p.GetPdgCode() == 321) {
        kaons.insert(std::round(p.Px() * 1000.) / 1000.);
      } else if (p.GetPdgCode() == 13) {
        fromBox++;
      }
    }
  }
  std::cout << "HYBRIDTEST hepmc_events=" << kaons.size() << " box_events=" << fromBox << std::endl;
  BOOST_CHECK((kaons == std::set<double>{1., 2., 3., 4., 5.})); // every HepMC event exactly once
  BOOST_CHECK_EQUAL(fromBox, 5);
  BOOST_CHECK_EQUAL(log.count("ran out of events"), 0);
}

/// A negative fraction is rejected when the configuration is read: the sequential mode would stay on that
/// generator for ever (its event counter never equals a negative fraction)
void scenarioNegativeFraction(ScenarioEnv const& env)
{
  LogCatcher log("scenario_negative_fraction", fair::Severity::error);
  BOOST_CHECK_THROW(startHybrid(env, hybridJson({boxJson(13, 1), boxJson(22, 1)}, "-1, 1"), 5), fair::FatalException);
  BOOST_CHECK_EQUAL(log.count("Fractions must be non-negative integers"), 1);
}

/// All fractions 0 is rejected also in random mode, where the first generator would silently be used for every
/// event (no probability is assigned to any of them)
void scenarioZeroFractionsRandom(ScenarioEnv const& env)
{
  LogCatcher log("scenario_zero_fractions_random", fair::Severity::error);
  BOOST_CHECK_THROW(startHybrid(env, hybridJson({boxJson(13, 1), boxJson(22, 1)}, "0, 0"), 5, 2, /*randomize=*/true),
                    fair::FatalException);
  BOOST_CHECK_EQUAL(log.count("All fractions provided are 0"), 1);
}

/// The job asks for one event more (10) than the pool holds (9): this must remain a fatal error - the job must
/// not end "successfully" with fewer events than requested. The LOG(fatal) comes from a worker thread, where it
/// cannot be turned into an exception which could be caught: the process ends with the exit code below.
void scenarioTooFewEvents(ScenarioEnv const& env)
{
  fair::Logger::OnFatal([] {
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(kFatalExitCode);
  });
  constexpr int nfiles = 3;
  auto poolDir = env.dir / "pool";
  createPool(poolDir, nfiles);
  const unsigned int nevents = poolEventsInFiles(nfiles) + 1;
  auto& hybrid = startHybrid(env, hybridJson({poolJson(poolDir.string())}, "1"), nevents);
  BOOST_REQUIRE(hybrid.Init());
  produce(hybrid, nevents); // does not come back
  BOOST_ERROR("the job went through although the pool holds fewer events than requested");
}

/// Fractions 9:9 over 18 events: the pool (9 events) serves the first 9 events of the job, the box the last 9.
/// The pool is used up in the middle of the job, so the guard on the last event of the job does not help: the
/// request budget must keep the hybrid from asking the pool for a 10th event while the box events are produced.
void scenarioMidJobShare(ScenarioEnv const& env)
{
  LogCatcher log("scenario_mid_job_share");
  auto poolFile = (env.dir / "pool" / "evtpool.root").string();
  fs::create_directories(env.dir / "pool");
  createKineFile(poolFile, 9, 0);
  const unsigned int nevents = 18;
  auto& hybrid = startHybrid(env, hybridJson({poolJson(poolFile), boxJson(13, 1)}, "9, 9"), nevents);
  BOOST_REQUIRE(hybrid.Init());
  BOOST_CHECK((expectedShares(hybrid) == std::vector<unsigned int>{9, 9}));

  auto events = produce(hybrid, nevents);
  settle();

  BOOST_REQUIRE_EQUAL(events.size(), nevents);
  for (unsigned int i = 0; i < nevents; ++i) {
    BOOST_CHECK_EQUAL(poolEventId(events[i]) >= 0., i < 9);
  }
  auto pool = subPool(hybrid, 0);
  std::cout << "HYBRIDTEST served=" << pool->getO2KineGenerator()->getEventsServed() << std::endl;
  BOOST_CHECK_EQUAL(pool->getO2KineGenerator()->getEventsServed(), 9);
  BOOST_CHECK_EQUAL(log.count("ran out of events"), 0);
}

/// Fractions 0:1: the pool is never used, so it is not asked for any event at all (it used to produce one in vain)
void scenarioZeroFraction(ScenarioEnv const& env)
{
  LogCatcher log("scenario_zero_fraction");
  auto poolFile = (env.dir / "pool" / "evtpool.root").string();
  fs::create_directories(env.dir / "pool");
  createKineFile(poolFile, 9, 0);
  const unsigned int nevents = 5;
  auto& hybrid = startHybrid(env, hybridJson({poolJson(poolFile), boxJson(13, 1)}, "0, 1"), nevents);
  BOOST_REQUIRE(hybrid.Init());
  BOOST_CHECK((expectedShares(hybrid) == std::vector<unsigned int>{0, 5}));

  auto events = produce(hybrid, nevents);
  settle();

  BOOST_REQUIRE_EQUAL(events.size(), nevents);
  for (auto const& e : events) {
    BOOST_CHECK_LT(poolEventId(e), 0.);
  }
  BOOST_CHECK_EQUAL(subPool(hybrid, 0)->getO2KineGenerator()->getEventsServed(), 0);
}

/// Random mode, two pools with fractions 1:1: pool A holds 9 events, pool B enough for the job. The seed is chosen
/// such that A is drawn exactly 9 times and the event after the 9th one comes from B: the lookahead asks A for a
/// 10th event, which fails in the worker thread. This event is never consumed, so the job must go through. Only
/// the main thread draws from gRandom here (the pools neither shuffle nor rotate), so the sequence is reproducible.
void scenarioUnusedFailure(ScenarioEnv const& env)
{
  LogCatcher log("scenario_unused_failure"); // LOG(fatal) throws, as in the simulation devices
  // keeps the job alive long enough after its last event for a worker to pick up the lookahead task of pool A
  RaceWindowWidener widener(env.windowMs);
  auto poolFileA = (env.dir / "poolA" / "evtpool.root").string();
  fs::create_directories(env.dir / "poolA");
  createKineFile(poolFileA, 9, 0);
  auto poolDirB = env.dir / "poolB";
  createPool(poolDirB, 5); // 20 events

  // the generator of each event is pool A if the draw is <= 0.5 (see GeneratorHybrid::generateEvent)
  constexpr int nA = 9;
  unsigned int nevents = 0;
  UInt_t seed = 0;
  while (nevents == 0) {
    gRandom->SetSeed(++seed);
    int a = 0;
    for (unsigned int i = 0; i < 25; ++i) { // at most 17 events from pool B
      if (gRandom->Rndm() <= 0.5 && ++a == nA) {
        if (gRandom->Rndm() > 0.5) {
          nevents = i + 2; // the 9th event of A, then one of B, which ends the job
        }
        break;
      }
    }
  }
  std::cout << "HYBRIDTEST seed=" << seed << " nevents=" << nevents << std::endl;
  auto json = hybridJson({poolJson(poolFileA), poolJson(poolDirB.string())}, "1, 1");
  auto& hybrid = startHybrid(env, json, nevents, 2, /*randomize=*/true);
  BOOST_REQUIRE(hybrid.Init());
  gRandom->SetSeed(seed);

  auto events = produce(hybrid, nevents);
  settle();

  BOOST_REQUIRE_EQUAL(events.size(), nevents);
  BOOST_CHECK_EQUAL(subPool(hybrid, 0)->getO2KineGenerator()->getEventsServed(), nA);
  // the lookahead did fail, but this did not matter
  BOOST_CHECK_EQUAL(log.count("ran out of events after 9 event(s)"), 1);
  BOOST_CHECK_EQUAL(log.count("failed to generate the requested event"), 0);
  BOOST_CHECK_MESSAGE(!widener.enabled() || widener.triggered(), "the race window was not entered: the scenario tests nothing");
}

/// As too_few_events, but LOG(fatal) throws (as in the simulation devices) and two worker threads run: the
/// failure of the pool in the worker thread must reach the main thread. Without the forwarding the exception
/// stays in the TBB worker, the other worker keeps polling and the job hangs (the watchdog of the child ends it).
void scenarioFailureForwarded(ScenarioEnv const& env)
{
  LogCatcher log("scenario_failure_forwarded", fair::Severity::error);
  constexpr int nfiles = 3;
  auto poolDir = env.dir / "pool";
  createPool(poolDir, nfiles);
  const unsigned int nevents = poolEventsInFiles(nfiles) + 1;
  auto& hybrid = startHybrid(env, hybridJson({poolJson(poolDir.string())}, "1"), nevents, 2);
  BOOST_REQUIRE(hybrid.Init());
  BOOST_CHECK_THROW(produce(hybrid, nevents), fair::FatalException);
  settle(); // the worker threads are detached: let them see the stop flag before the process ends
  BOOST_CHECK_EQUAL(log.count("ran out of events after 9 event(s) from 3 input file(s) (10 were requested)"), 1);
  BOOST_CHECK_EQUAL(log.count("Sub-generator 0 (evtpool) failed to generate the requested event"), 1);
}

std::map<std::string, std::function<void(ScenarioEnv const&)>> const& scenarios()
{
  static std::map<std::string, std::function<void(ScenarioEnv const&)>> const s = {
    {"exact_pool", scenarioExactPool},
    {"larger_pool", [](ScenarioEnv const& env) { scenarioLargerPool(env, false); }},
    {"larger_pool_random", [](ScenarioEnv const& env) { scenarioLargerPool(env, true); }},
    {"cocktail_pool", scenarioCocktailPool},
    {"cocktail_groups", scenarioCocktailGroups},
    {"parallel_mode", scenarioParallelMode},
    {"mixed_sequence", scenarioMixedSequence},
    {"uneven_shares", scenarioUnevenShares},
    {"random_shares", scenarioRandomShares},
    {"init_failure", scenarioInitFailure},
    {"init_failure_hepmc", scenarioInitFailureHepMC},
    {"hepmc_share", scenarioHepMCShare},
    {"negative_fraction", scenarioNegativeFraction},
    {"zero_fractions_random", scenarioZeroFractionsRandom},
    {"too_few_events", scenarioTooFewEvents},
    {"mid_job_share", scenarioMidJobShare},
    {"zero_fraction", scenarioZeroFraction},
    {"unused_failure", scenarioUnusedFailure},
    {"failure_forwarded", scenarioFailureForwarded}};
  return s;
}

// ---------------------------------------------------------------------------------------------------------------
// helpers of the parent process
// ---------------------------------------------------------------------------------------------------------------

struct ChildResult {
  bool exited = false;
  int status = -1; // exit code, or the number of the signal which ended the child
  std::string output;

  /// the interesting part of the output of the child, for the failure messages
  std::string digest() const
  {
    std::istringstream in(output);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
      lines.push_back(line);
    }
    std::string d;
    for (size_t i = 0; i < lines.size(); ++i) {
      bool interesting = lines[i].find("HYBRIDTEST") != std::string::npos || lines[i].find("error") != std::string::npos ||
                         lines[i].find("fatal") != std::string::npos || lines[i].find("FATAL") != std::string::npos ||
                         lines[i].find("warn") != std::string::npos || lines[i].find("WARN") != std::string::npos ||
                         i + 25 >= lines.size();
      if (interesting) {
        d += lines[i] + "\n";
      }
    }
    return std::string("child ") + (exited ? "exited with code " : "was killed by signal ") + std::to_string(status) + "\n" + d;
  }
};

/// starts this executable again, with the given scenario
ChildResult runChild(std::string const& scenario, int windowMs = kDefaultWindowMs)
{
  TempDir tmp("hybrid_" + scenario);
  auto logFile = (tmp.path / "child.log").string();
  auto exe = std::string(boost::unit_test::framework::master_test_suite().argv[0]);
  setenv(kScenarioEnv, scenario.c_str(), 1);
  setenv(kDirEnv, (tmp.path / "work").c_str(), 1);
  setenv(kWindowEnv, std::to_string(windowMs).c_str(), 1);
  // "exec": the shell is replaced by the child, so that a signal ending it is not hidden in an exit code
  auto cmd = "exec \"" + exe + "\" --run_test=HybridScenarioWorker > \"" + logFile + "\" 2>&1";
  int rc = std::system(cmd.c_str());
  unsetenv(kScenarioEnv);
  unsetenv(kDirEnv);
  unsetenv(kWindowEnv);

  ChildResult result;
  result.exited = WIFEXITED(rc);
  result.status = WIFEXITED(rc) ? WEXITSTATUS(rc) : (WIFSIGNALED(rc) ? WTERMSIG(rc) : -1);
  std::ifstream in(logFile);
  std::stringstream ss;
  ss << in.rdbuf();
  result.output = ss.str();
  return result;
}

void expectSuccess(std::string const& scenario)
{
  auto result = runChild(scenario);
  BOOST_CHECK_MESSAGE(result.exited && result.status == 0, "scenario '" << scenario << "' failed: " << result.digest());
}

} // namespace

// Finding 2: no lookahead after the last event of the job ----------------------------------------------------------

BOOST_AUTO_TEST_CASE(Hybrid_ExactPool_NoLookaheadAfterLastEvent)
{
  expectSuccess("exact_pool");
}

BOOST_AUTO_TEST_CASE(Hybrid_LargerPool_NoEventBeyondTheJob)
{
  expectSuccess("larger_pool");
}

BOOST_AUTO_TEST_CASE(Hybrid_LargerPool_RandomMode_NoEventBeyondTheJob)
{
  expectSuccess("larger_pool_random");
}

BOOST_AUTO_TEST_CASE(Hybrid_Cocktail_ExactPool)
{
  expectSuccess("cocktail_pool");
}

// the other operation modes ------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(Hybrid_Cocktail_Groups)
{
  expectSuccess("cocktail_groups");
}

BOOST_AUTO_TEST_CASE(Hybrid_ParallelMode)
{
  expectSuccess("parallel_mode");
}

// Findings 4/8: expectations and initialisation of the sub-generators ------------------------------------------------

BOOST_AUTO_TEST_CASE(Hybrid_SubGeneratorDoesNotAssumeTheWholeJob)
{
  expectSuccess("mixed_sequence");
}

BOOST_AUTO_TEST_CASE(Hybrid_SubGeneratorShare_FromFractions)
{
  auto result = runChild("uneven_shares");
  BOOST_CHECK_MESSAGE(result.exited && result.status == kFatalExitCode, "expected the fatal exit of the child: " << result.digest());
  BOOST_CHECK_MESSAGE(result.output.find("HYBRIDTEST shares=7,3, kine=7") != std::string::npos,
                      "the shares computed from the fractions are wrong: " << result.digest());
  BOOST_CHECK_MESSAGE(result.output.find("This job will request 7 events, but the input (1 file(s), 6 events in the first one) holds only about 6") != std::string::npos,
                      "the pool did not warn about its share: " << result.digest());
  BOOST_CHECK_MESSAGE(result.output.find("ran out of events after 6 event(s) from 1 input file(s) (7 were requested)") != std::string::npos,
                      "the failure does not report the share of the pool: " << result.digest());
}

BOOST_AUTO_TEST_CASE(Hybrid_SubGeneratorShare_Randomized)
{
  expectSuccess("random_shares");
}

BOOST_AUTO_TEST_CASE(Hybrid_SubGeneratorInitFailureIsFatal)
{
  expectSuccess("init_failure");
}

BOOST_AUTO_TEST_CASE(Hybrid_SubGeneratorInitResultIsChecked)
{
  expectSuccess("init_failure_hepmc");
}

BOOST_AUTO_TEST_CASE(Hybrid_HepMCSubGenerator_KnowsItsShare)
{
  expectSuccess("hepmc_share");
}

BOOST_AUTO_TEST_CASE(Hybrid_NegativeFraction_IsRejected)
{
  expectSuccess("negative_fraction");
}

BOOST_AUTO_TEST_CASE(Hybrid_ZeroFractionsRandom_IsRejected)
{
  expectSuccess("zero_fractions_random");
}

// Finding 3: a pool which is too small remains a fatal error ---------------------------------------------------------

BOOST_AUTO_TEST_CASE(Hybrid_PoolTooSmall_IsFatal)
{
  auto result = runChild("too_few_events");
  BOOST_CHECK_MESSAGE(result.exited && result.status == kFatalExitCode, "expected the fatal exit of the child: " << result.digest());
  BOOST_CHECK_MESSAGE(result.output.find("ran out of events after 9 event(s) from 3 input file(s)") != std::string::npos,
                      "the reason of the failure is not reported: " << result.digest());
  // the only generator of the hybrid is asked for all the events of the job, and knows it
  BOOST_CHECK_MESSAGE(result.output.find("ran out of events after 9 event(s) from 3 input file(s) (10 were requested)") != std::string::npos,
                      "the number of requested events is not reported: " << result.digest());
}

// Follow-up of finding 2: request budget, failures in the worker threads ----------------------------------------------

BOOST_AUTO_TEST_CASE(Hybrid_PoolUsedUpMidJob_NotAskedBeyondItsShare)
{
  expectSuccess("mid_job_share");
}

BOOST_AUTO_TEST_CASE(Hybrid_ZeroFraction_NeverAsked)
{
  expectSuccess("zero_fraction");
}

BOOST_AUTO_TEST_CASE(Hybrid_UnconsumedLookaheadFailure_IsHarmless)
{
  expectSuccess("unused_failure");
}

BOOST_AUTO_TEST_CASE(Hybrid_WorkerFailure_ReachesMainThread)
{
  expectSuccess("failure_forwarded");
}

// the child process --------------------------------------------------------------------------------------------

/// Runs the scenario given by the environment; started by the test cases above, a no-op otherwise.
BOOST_AUTO_TEST_CASE(HybridScenarioWorker)
{
  const char* scenario = std::getenv(kScenarioEnv);
  if (!scenario) {
    return;
  }
  // no core dumps from the scenarios which fail, and no hanging forever if something goes wrong; a watchdog
  // thread rather than alarm(): Boost.Test handles SIGALRM itself, and a hung hybrid then blocks the exit
  rlimit noCore{0, 0};
  setrlimit(RLIMIT_CORE, &noCore);
  std::thread([] {
    std::this_thread::sleep_for(std::chrono::seconds(kWatchdogSeconds));
    std::cout << "HYBRIDTEST watchdog: the scenario did not end within " << kWatchdogSeconds << " s" << std::endl;
    std::_Exit(kWatchdogExitCode);
  }).detach();

  auto it = scenarios().find(scenario);
  BOOST_REQUIRE_MESSAGE(it != scenarios().end(), "unknown scenario " << scenario);
  ScenarioEnv env;
  env.dir = std::getenv(kDirEnv) ? std::getenv(kDirEnv) : "hybrid_scenario";
  env.windowMs = std::getenv(kWindowEnv) ? std::atoi(std::getenv(kWindowEnv)) : kDefaultWindowMs;
  fs::create_directories(env.dir);
  it->second(env);
}
