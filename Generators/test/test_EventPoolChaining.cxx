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

/// \file test_EventPoolChaining.cxx
/// \brief tests reading several (event pool) kinematics files one after the other
///        in GeneratorFromO2Kine and GeneratorFromEventPool
/// \author M. Giacalone, mgiacalo@cern.ch, 08/2026

#define BOOST_TEST_MODULE Test EventPoolChaining
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <fairlogger/Logger.h>

#include <Generators/GeneratorFromFile.h>
#include <Generators/GeneratorFromO2KineParam.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/MCTrack.h>

#include <TFile.h>
#include <TROOT.h>
#include <TTree.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace
{

/// the px of the first track of an event encodes the file it belongs to and its
/// position within that file; this allows to check the reading order later on
double encodeMomentum(int fileTag, int event)
{
  return 1000. * fileTag + event;
}

/// creates a minimal - but structurally valid - O2 kinematics file;
/// event `ev` contains `ev + 1` primary tracks
void createKineFile(std::string const& path, int nevents, int fileTag)
{
  std::unique_ptr<TFile> file(TFile::Open(path.c_str(), "RECREATE"));
  auto tree = new TTree("o2sim", "o2sim"); // owned by the file
  std::vector<o2::MCTrack> tracks;
  tree->Branch("MCTrack", &tracks);
  o2::dataformats::MCEventHeader header;
  auto headerPtr = &header;
  tree->Branch("MCEventHeader.", &headerPtr);

  for (int ev = 0; ev < nevents; ++ev) {
    tracks.clear();
    for (int i = 0; i <= ev; ++i) {
      // a primary track has no mothers
      o2::MCTrack track(211, -1, -1, -1, -1, encodeMomentum(fileTag, ev), 0., 0., 0., 0., 0., 0., 0);
      track.setToBeDone(true);
      tracks.push_back(track);
    }
    header.Reset();
    header.SetEventID(static_cast<int>(encodeMomentum(fileTag, ev)));
    header.SetVertex(0., 0., 0.);
    header.putInfo<int>("test_fileTag", fileTag);
    tree->Fill();
  }
  tree->Write();
  file->Close();
}

/// creates `nfiles` event pool files under `<tmpDir>/<i>/evtpool.root`;
/// file i holds i + 2 events
std::vector<std::string> createPool(fs::path const& tmpDir, int nfiles)
{
  std::vector<std::string> filenames;
  for (int i = 0; i < nfiles; ++i) {
    auto fileDir = tmpDir / std::to_string(i);
    fs::create_directories(fileDir);
    auto filePath = fileDir / o2::eventgen::GeneratorFromEventPool::eventpool_filename;
    createKineFile(filePath.string(), i + 2, i);
    filenames.push_back(filePath.string());
  }
  return filenames;
}

/// number of events in pool file i (must match createPool)
int eventsInFile(int i) { return i + 2; }

/// scratch directory that removes itself again; it has to be declared before the
/// generators using it, so that the generators (and with them the open files) are
/// destructed first
struct TempDir {
  explicit TempDir(std::string const& tag)
  {
    path = fs::temp_directory_path() / (tag + "_" + std::to_string(getpid()) + "_" + std::to_string(std::rand()));
    std::error_code ec;
    fs::remove_all(path, ec);
    fs::create_directories(path);
  }
  ~TempDir()
  {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
  fs::path path;
};

/// number of ROOT files currently open in the process
int openRootFiles()
{
  return gROOT->GetListOfFiles() ? gROOT->GetListOfFiles()->GetEntries() : 0;
}

/// reads the next event and returns the identifier encoded in the first track
double readNextEvent(o2::eventgen::Generator& gen)
{
  gen.clearParticles();
  if (!gen.importParticles()) {
    return -1.;
  }
  auto const& particles = gen.getParticles();
  if (particles.empty()) {
    return -1.;
  }
  return particles.front().Px();
}

} // namespace

/// several files are read one after the other, in the order in which they were given
BOOST_AUTO_TEST_CASE(Rollover_MultipleFiles_Sequential)
{
  TempDir tmpDirGuard("rollover_sequential");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 3;
  auto filenames = createPool(tmpDir, numfiles);

  o2::eventgen::GeneratorFromO2Kine gen(filenames);
  BOOST_CHECK_EQUAL(gen.getNumberOfFiles(), numfiles);
  BOOST_CHECK(gen.Init());
  // only the first file is known/open at this point
  BOOST_CHECK_EQUAL(gen.getEventsAvailable(), eventsInFile(0));
  BOOST_CHECK_EQUAL(gen.getCurrentFileIndex(), 0);

  // events must come out file by file, in order
  for (int file = 0; file < numfiles; ++file) {
    for (int ev = 0; ev < eventsInFile(file); ++ev) {
      BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(file, ev), 1E-6);
      BOOST_CHECK_EQUAL(gen.getCurrentFileIndex(), file);
    }
  }
  // all files are used up now; asking for more is a fatal condition, which cannot be
  // checked from within the process (see the example script for the end-to-end check)
  BOOST_CHECK_EQUAL(gen.getNumberOfFilesUsed(), numfiles);
}

/// the files must be opened lazily: only when the events of the current one are
/// exhausted, and never more than one at a time
BOOST_AUTO_TEST_CASE(Rollover_OpensFilesLazily)
{
  TempDir tmpDirGuard("rollover_lazy");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 4;
  auto filenames = createPool(tmpDir, numfiles);

  auto filesOpenBefore = openRootFiles();

  o2::eventgen::GeneratorFromO2Kine gen(filenames);
  // the constructor must not open anything at all
  BOOST_CHECK_EQUAL(openRootFiles() - filesOpenBefore, 0);
  BOOST_CHECK(gen.Init());
  // exactly one input file is open, no matter how many were given
  BOOST_CHECK_EQUAL(openRootFiles() - filesOpenBefore, 1);
  BOOST_CHECK_EQUAL(gen.getNumberOfFilesUsed(), 1);

  // reading the events of the first file must not touch any other file
  for (int ev = 0; ev < eventsInFile(0); ++ev) {
    readNextEvent(gen);
    BOOST_CHECK_EQUAL(gen.getNumberOfFilesUsed(), 1);
    BOOST_CHECK_EQUAL(openRootFiles() - filesOpenBefore, 1);
  }

  // the next event triggers opening the second file - and only the second one
  readNextEvent(gen);
  BOOST_CHECK_EQUAL(gen.getCurrentFileIndex(), 1);
  BOOST_CHECK_EQUAL(gen.getNumberOfFilesUsed(), 2);
  BOOST_CHECK_EQUAL(openRootFiles() - filesOpenBefore, 1);
}

/// a file which cannot be read is only discovered - and skipped - when it is reached
BOOST_AUTO_TEST_CASE(Rollover_SkipsBadFilesLazily)
{
  TempDir tmpDirGuard("rollover_badfiles");
  auto const& tmpDir = tmpDirGuard.path;
  auto good0 = (tmpDir / "good0.root").string();
  auto good1 = (tmpDir / "good1.root").string();
  createKineFile(good0, 2, 0);
  createKineFile(good1, 2, 1);
  auto nonexisting = (tmpDir / "doesnotexist.root").string();

  // the broken file sits in the middle of the list: construction must succeed
  o2::eventgen::GeneratorFromO2Kine gen({good0, nonexisting, good1});
  BOOST_CHECK_EQUAL(gen.getNumberOfFiles(), 3);
  BOOST_CHECK(gen.Init());
  BOOST_CHECK_EQUAL(gen.getCurrentFileIndex(), 0);

  // the two events of the first file, then the broken one is skipped
  BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(0, 0), 1E-6);
  BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(0, 1), 1E-6);
  BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(1, 0), 1E-6);
  BOOST_CHECK_EQUAL(gen.getCurrentFileIndex(), 2);

  // nothing usable at all -> Init must fail rather than crash
  o2::eventgen::GeneratorFromO2Kine badgen({nonexisting});
  BOOST_CHECK(!badgen.Init());
}

/// the number of particles per event must be preserved across the file boundaries
BOOST_AUTO_TEST_CASE(Rollover_ParticleContent)
{
  TempDir tmpDirGuard("rollover_content");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 3;
  auto filenames = createPool(tmpDir, numfiles);

  o2::eventgen::GeneratorFromO2Kine gen(filenames);
  BOOST_CHECK(gen.Init());

  for (int file = 0; file < numfiles; ++file) {
    for (int ev = 0; ev < eventsInFile(file); ++ev) {
      gen.clearParticles();
      BOOST_CHECK(gen.importParticles());
      BOOST_CHECK_EQUAL(gen.getParticles().size(), static_cast<size_t>(ev + 1));
    }
  }
}

/// the MC event header of the original file must be forwarded also when reading
/// from a file that is not the first one of the list
BOOST_AUTO_TEST_CASE(Rollover_HeaderForwarding)
{
  TempDir tmpDirGuard("rollover_header");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 3;
  auto filenames = createPool(tmpDir, numfiles);

  o2::eventgen::GeneratorFromO2Kine gen(filenames);
  BOOST_CHECK(gen.Init());

  for (int file = 0; file < numfiles; ++file) {
    for (int ev = 0; ev < eventsInFile(file); ++ev) {
      gen.clearParticles();
      BOOST_CHECK(gen.importParticles());

      o2::dataformats::MCEventHeader header;
      gen.updateHeader(&header);
      BOOST_CHECK_EQUAL(static_cast<int>(header.GetEventID()), static_cast<int>(encodeMomentum(file, ev)));

      bool isvalid = false;
      auto tag = header.getInfo<int>("test_fileTag", isvalid);
      BOOST_CHECK(isvalid);
      BOOST_CHECK_EQUAL(tag, file);

      // the bookkeeping information must point to the file the event was read from
      auto inputFile = header.getInfo<std::string>("forwarding-generator_inputFile", isvalid);
      BOOST_CHECK(isvalid);
      BOOST_CHECK_EQUAL(inputFile, filenames[file]);

      // ... and to the entry within that very file
      auto entry = header.getInfo<int>("forwarding-generator_inputEventNumber", isvalid);
      BOOST_CHECK(isvalid);
      BOOST_CHECK_EQUAL(entry, ev);
    }
  }
}

/// a comma-separated list of file names is read one file after the other as well
BOOST_AUTO_TEST_CASE(Rollover_CommaSeparatedFileNames)
{
  TempDir tmpDirGuard("rollover_commalist");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 3;
  auto filenames = createPool(tmpDir, numfiles);

  std::string joined;
  for (auto const& f : filenames) {
    joined += (joined.empty() ? "" : ",") + f;
  }

  auto splitted = o2::eventgen::GeneratorFromO2Kine::splitFileNames(joined);
  BOOST_CHECK_EQUAL(splitted.size(), static_cast<size_t>(numfiles));
  // white space around the separators must be tolerated
  BOOST_CHECK_EQUAL(o2::eventgen::GeneratorFromO2Kine::splitFileNames(" a.root , b.root ,,").size(), 2u);

  o2::eventgen::GeneratorFromO2Kine gen(joined.c_str());
  BOOST_CHECK_EQUAL(gen.getNumberOfFiles(), numfiles);
  BOOST_CHECK(gen.Init());
  for (int file = 0; file < numfiles; ++file) {
    for (int ev = 0; ev < eventsInFile(file); ++ev) {
      BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(file, ev), 1E-6);
    }
  }
}

/// round robin must wrap around the whole file list, not around a single file
BOOST_AUTO_TEST_CASE(Rollover_RoundRobin)
{
  TempDir tmpDirGuard("rollover_roundrobin");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 2;
  auto filenames = createPool(tmpDir, numfiles);
  const int total = eventsInFile(0) + eventsInFile(1);

  o2::eventgen::O2KineGenConfig config;
  config.roundRobin = true;
  o2::eventgen::GeneratorFromO2Kine gen(config, filenames);
  BOOST_CHECK(gen.Init());

  // read twice as many events as available; the second pass must repeat the first one
  std::vector<double> firstPass;
  for (int i = 0; i < total; ++i) {
    firstPass.push_back(readNextEvent(gen));
  }
  for (int i = 0; i < total; ++i) {
    BOOST_CHECK_CLOSE(readNextEvent(gen), firstPass[i], 1E-6);
  }
}

/// a single file, read without round robin, must NOT be silently reopened/reused once
/// exhausted: openNextFile()'s "next == mCurrentFileIndex && mCurrentFile" shortcut
/// (which restarts the currently open file in place) may only ever fire when wrapAround
/// (i.e. roundRobin) is true; with roundRobin off, running out of events is fatal
BOOST_AUTO_TEST_CASE(SingleFile_NoRoundRobin_ExhaustionIsFatal)
{
  TempDir tmpDirGuard("single_file_no_rr");
  auto const& tmpDir = tmpDirGuard.path;
  auto file = (tmpDir / "kine.root").string();
  constexpr int nevents = 3;
  createKineFile(file, nevents, 0);

  o2::eventgen::O2KineGenConfig config;
  config.roundRobin = false;
  o2::eventgen::GeneratorFromO2Kine gen(config, {file});
  BOOST_CHECK(gen.Init());

  // all events of the single file are served normally, without any repeats
  std::set<double> seen;
  for (int i = 0; i < nevents; ++i) {
    auto id = readNextEvent(gen);
    BOOST_CHECK(id >= 0.);
    BOOST_CHECK(seen.insert(id).second);
  }
  BOOST_CHECK_EQUAL(seen.size(), static_cast<size_t>(nevents));

  // the next request must crash the job (fatal), not silently restart the same file
  BOOST_CHECK_THROW(readNextEvent(gen), fair::FatalException);
}

/// the same single-file setup, but with roundRobin enabled: the already-open file must
/// be reused in place (no reopen), giving a fresh pass of the very same events
BOOST_AUTO_TEST_CASE(SingleFile_RoundRobin_ReusesWithoutReopening)
{
  TempDir tmpDirGuard("single_file_rr");
  auto const& tmpDir = tmpDirGuard.path;
  auto file = (tmpDir / "kine.root").string();
  constexpr int nevents = 3;
  createKineFile(file, nevents, 0);

  auto filesOpenBefore = openRootFiles();

  o2::eventgen::O2KineGenConfig config;
  config.roundRobin = true;
  o2::eventgen::GeneratorFromO2Kine gen(config, {file});
  BOOST_CHECK(gen.Init());
  BOOST_CHECK_EQUAL(openRootFiles() - filesOpenBefore, 1);

  std::vector<double> firstPass;
  for (int i = 0; i < nevents; ++i) {
    firstPass.push_back(readNextEvent(gen));
  }
  // wrapping around must not close and reopen the file
  for (int i = 0; i < nevents; ++i) {
    BOOST_CHECK_CLOSE(readNextEvent(gen), firstPass[i], 1E-6);
    BOOST_CHECK_EQUAL(openRootFiles() - filesOpenBefore, 1);
  }
  BOOST_CHECK_EQUAL(gen.getNumberOfFilesUsed(), 1);
}

/// round robin must serve every event of every file, in order, and only then start
/// over with the first file again - for an arbitrary number of passes
BOOST_AUTO_TEST_CASE(Rollover_RoundRobin_FullPasses)
{
  TempDir tmpDirGuard("rollover_rr_full");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 3;
  auto filenames = createPool(tmpDir, numfiles);

  auto filesOpenBefore = openRootFiles();

  o2::eventgen::O2KineGenConfig config;
  config.roundRobin = true;
  o2::eventgen::GeneratorFromO2Kine gen(config, filenames);
  BOOST_CHECK(gen.Init());

  constexpr int npasses = 3;
  for (int pass = 0; pass < npasses; ++pass) {
    for (int file = 0; file < numfiles; ++file) {
      for (int ev = 0; ev < eventsInFile(file); ++ev) {
        // the very same sequence must come back on every pass
        BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(file, ev), 1E-6);
        BOOST_CHECK_EQUAL(gen.getCurrentFileIndex(), file);
        BOOST_CHECK_EQUAL(gen.getEventsAvailable(), eventsInFile(file));
        // laziness is not given up when wrapping around
        BOOST_CHECK_EQUAL(openRootFiles() - filesOpenBefore, 1);
      }
    }
  }
  // every file was opened exactly once per pass
  BOOST_CHECK_EQUAL(gen.getNumberOfFilesUsed(), numfiles * npasses);
}

/// with randomization each round robin pass must again contain every event exactly
/// once, but in a freshly drawn order
BOOST_AUTO_TEST_CASE(Rollover_RoundRobin_RandomizedPasses)
{
  TempDir tmpDirGuard("rollover_rr_random");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 3;
  auto filenames = createPool(tmpDir, numfiles);

  int total = 0;
  std::set<double> allEvents;
  for (int file = 0; file < numfiles; ++file) {
    total += eventsInFile(file);
    for (int ev = 0; ev < eventsInFile(file); ++ev) {
      allEvents.insert(encodeMomentum(file, ev));
    }
  }

  o2::eventgen::O2KineGenConfig config;
  config.roundRobin = true;
  config.randomize = true;
  config.rngseed = 99;
  o2::eventgen::GeneratorFromO2Kine gen(config, filenames);
  BOOST_CHECK(gen.Init());

  constexpr int npasses = 4;
  std::set<std::vector<double>> passOrders;
  for (int pass = 0; pass < npasses; ++pass) {
    std::set<double> seen;
    std::vector<double> order;
    for (int i = 0; i < total; ++i) {
      auto id = readNextEvent(gen);
      BOOST_CHECK(seen.insert(id).second); // no event twice within a pass
      order.push_back(id);
    }
    // a full pass covers the whole input, no more and no less
    BOOST_CHECK(seen == allEvents);
    passOrders.insert(order);
  }
  // the passes must not all come out in the very same order
  BOOST_CHECK(passOrders.size() > 1);
}

/// the generator knows, through Generator::gTotalNEvents, how many events the job is
/// going to ask for, and counts how many it has actually served
BOOST_AUTO_TEST_CASE(Rollover_AccountsForRequestedEvents)
{
  TempDir tmpDirGuard("rollover_accounting");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 3;
  auto filenames = createPool(tmpDir, numfiles);
  int total = 0;
  for (int i = 0; i < numfiles; ++i) {
    total += eventsInFile(i);
  }

  // this is what o2-sim / o2-sim-dpl-eventgen do before creating the generators
  unsigned int requested = total;
  o2::eventgen::Generator::setTotalNEvents(requested);
  BOOST_CHECK_EQUAL(o2::eventgen::Generator::getTotalNEvents(), static_cast<unsigned int>(total));

  o2::eventgen::GeneratorFromO2Kine gen(filenames);
  BOOST_CHECK(gen.Init());
  BOOST_CHECK_EQUAL(gen.getEventsServed(), 0);

  for (int i = 0; i < total; ++i) {
    BOOST_CHECK(readNextEvent(gen) >= 0.);
    // the counter runs over the whole input, not per file
    BOOST_CHECK_EQUAL(gen.getEventsServed(), i + 1);
  }
  BOOST_CHECK_EQUAL(gen.getEventsServed(), total);

  unsigned int reset = 0;
  o2::eventgen::Generator::setTotalNEvents(reset);
}

/// the event pool generator goes through the whole pool, one file after the other
BOOST_AUTO_TEST_CASE(EventPool_RollsOverAllFiles)
{
  TempDir tmpDirGuard("evtpool_rollover");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 5;
  createPool(tmpDir, numfiles);

  int expectedEvents = 0;
  for (int i = 0; i < numfiles; ++i) {
    expectedEvents += eventsInFile(i);
  }

  auto filesOpenBefore = openRootFiles();

  o2::eventgen::EventPoolGenConfig config;
  config.eventPoolPath = tmpDir.string();
  config.randomize = false;
  config.rngseed = 42;
  o2::eventgen::GeneratorFromEventPool gen(config);
  BOOST_CHECK(gen.Init());
  BOOST_CHECK_EQUAL(gen.getFileUniverse().size(), static_cast<size_t>(numfiles));
  BOOST_CHECK_EQUAL(gen.getChosenFiles().size(), static_cast<size_t>(numfiles));
  BOOST_CHECK_EQUAL(gen.getO2KineGenerator()->getNumberOfFiles(), numfiles);
  // still only one file open, whatever the size of the pool
  BOOST_CHECK_EQUAL(gen.getO2KineGenerator()->getNumberOfFilesUsed(), 1);
  BOOST_CHECK_EQUAL(openRootFiles() - filesOpenBefore, 1);

  // every single event of the pool must be delivered exactly once
  std::set<double> seen;
  for (int i = 0; i < expectedEvents; ++i) {
    auto id = readNextEvent(gen);
    BOOST_CHECK(id >= 0.);
    BOOST_CHECK(seen.insert(id).second); // no duplicates
    BOOST_CHECK_EQUAL(openRootFiles() - filesOpenBefore, 1);
  }
  BOOST_CHECK_EQUAL(seen.size(), static_cast<size_t>(expectedEvents));
  BOOST_CHECK_EQUAL(gen.getO2KineGenerator()->getNumberOfFilesUsed(), numfiles);
}

/// the order in which the pool files are visited must be reproducible for a given seed
BOOST_AUTO_TEST_CASE(EventPool_SelectionIsReproducible)
{
  TempDir tmpDirGuard("evtpool_selection");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 8;
  createPool(tmpDir, numfiles);

  auto orderFor = [&tmpDir](unsigned int seed) {
    o2::eventgen::EventPoolGenConfig config;
    config.eventPoolPath = tmpDir.string();
    config.rngseed = seed;
    o2::eventgen::GeneratorFromEventPool gen(config);
    gen.Init();
    return gen.getChosenFiles();
  };

  // the same seed always gives the same order, and every file is included
  auto a = orderFor(1);
  auto b = orderFor(1);
  BOOST_CHECK_EQUAL(a.size(), static_cast<size_t>(numfiles));
  BOOST_CHECK(a == b);

  // ... while different seeds do not all collapse onto the same order
  std::set<std::vector<std::string>> orders;
  for (unsigned int seed = 1; seed <= 10; ++seed) {
    orders.insert(orderFor(seed));
  }
  BOOST_CHECK(orders.size() > 1);
}

/// with randomization every event of the pool is still served exactly once: the order
/// within a file is a permutation of its entries, fixed when the file is opened
BOOST_AUTO_TEST_CASE(EventPool_RandomizeIsAPermutation)
{
  TempDir tmpDirGuard("evtpool_randomize");
  auto const& tmpDir = tmpDirGuard.path;
  constexpr int numfiles = 4;
  createPool(tmpDir, numfiles);

  int expectedEvents = 0;
  for (int i = 0; i < numfiles; ++i) {
    expectedEvents += eventsInFile(i);
  }

  auto filesOpenBefore = openRootFiles();

  o2::eventgen::EventPoolGenConfig config;
  config.eventPoolPath = tmpDir.string();
  config.randomize = true;
  config.rngseed = 12345;
  o2::eventgen::GeneratorFromEventPool gen(config);
  BOOST_CHECK(gen.Init());

  std::set<double> seen;
  std::vector<double> order;
  for (int i = 0; i < expectedEvents; ++i) {
    auto id = readNextEvent(gen);
    BOOST_CHECK(id >= 0.);
    // no event is served twice ...
    BOOST_CHECK(seen.insert(id).second);
    order.push_back(id);
    // ... and still only one file is open
    BOOST_CHECK_EQUAL(openRootFiles() - filesOpenBefore, 1);
  }
  BOOST_CHECK_EQUAL(seen.size(), static_cast<size_t>(expectedEvents));
  BOOST_CHECK_EQUAL(gen.getO2KineGenerator()->getNumberOfFilesUsed(), numfiles);

  // the order must actually differ from the sequential one
  auto sorted = order;
  std::sort(sorted.begin(), sorted.end());
  BOOST_CHECK(order != sorted);
}

/// randomization is a permutation also for a single file, and round robin gives a
/// fresh permutation on every pass
BOOST_AUTO_TEST_CASE(SingleFile_RandomizeRoundRobin)
{
  TempDir tmpDirGuard("single_randomize");
  auto const& tmpDir = tmpDirGuard.path;
  auto file = (tmpDir / "kine.root").string();
  constexpr int nevents = 6;
  createKineFile(file, nevents, 0);

  o2::eventgen::O2KineGenConfig config;
  config.randomize = true;
  config.roundRobin = true;
  config.rngseed = 7;
  o2::eventgen::GeneratorFromO2Kine gen(config, {file});
  BOOST_CHECK(gen.Init());

  // first pass: every event exactly once
  std::set<double> firstPass;
  std::vector<double> firstOrder;
  for (int i = 0; i < nevents; ++i) {
    auto id = readNextEvent(gen);
    BOOST_CHECK(firstPass.insert(id).second);
    firstOrder.push_back(id);
  }
  BOOST_CHECK_EQUAL(firstPass.size(), static_cast<size_t>(nevents));

  // second pass: same events again, but re-shuffled and without reopening the file
  std::set<double> secondPass;
  std::vector<double> secondOrder;
  for (int i = 0; i < nevents; ++i) {
    auto id = readNextEvent(gen);
    BOOST_CHECK(secondPass.insert(id).second);
    secondOrder.push_back(id);
  }
  BOOST_CHECK(firstPass == secondPass);
  BOOST_CHECK(firstOrder != secondOrder);
  BOOST_CHECK_EQUAL(gen.getNumberOfFilesUsed(), 1);
}
