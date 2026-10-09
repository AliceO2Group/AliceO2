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

#include "EventPoolTestUtils.h"
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
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

using namespace evtpooltest;

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

  // nothing usable at all -> Init must stop the job (LOG(fatal)), rather than fail silently and crash later:
  // FairPrimaryGenerator ignores the result of Init()
  LogCatcher log("badfiles_init", fair::Severity::fatal);
  o2::eventgen::GeneratorFromO2Kine badgen({nonexisting});
  BOOST_CHECK_THROW(badgen.Init(), fair::FatalException);
  BOOST_CHECK_EQUAL(log.count("none of the 1 input file(s) can be used"), 1);
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

/// the event header of one file must not be forwarded for the events of the next file,
/// in case that one carries no MCEventHeader at all
BOOST_AUTO_TEST_CASE(Rollover_NoStaleHeaderAcrossFiles)
{
  TempDir tmpDirGuard("rollover_staleheader");
  auto const& tmpDir = tmpDirGuard.path;
  auto withHeader = (tmpDir / "withheader.root").string();
  auto noHeader = (tmpDir / "noheader.root").string();
  createKineFile(withHeader, 2, 7);
  createKineFile(noHeader, 2, 8, /*withHeader=*/false);

  o2::eventgen::GeneratorFromO2Kine gen({withHeader, noHeader});
  BOOST_CHECK(gen.Init());

  bool isvalid = false;
  for (int ev = 0; ev < 2; ++ev) {
    BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(7, ev), 1E-6);
    o2::dataformats::MCEventHeader header;
    gen.updateHeader(&header);
    BOOST_CHECK_EQUAL(header.getInfo<int>("test_fileTag", isvalid), 7);
    BOOST_CHECK(isvalid);
  }
  for (int ev = 0; ev < 2; ++ev) {
    BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(8, ev), 1E-6);
    o2::dataformats::MCEventHeader header;
    gen.updateHeader(&header);
    // nothing of the first file may show up here
    header.getInfo<int>("test_fileTag", isvalid);
    BOOST_CHECK(!isvalid);
    BOOST_CHECK(static_cast<int>(header.GetEventID()) != static_cast<int>(encodeMomentum(7, 0)));
    BOOST_CHECK(static_cast<int>(header.GetEventID()) != static_cast<int>(encodeMomentum(7, 1)));
    // the bookkeeping still refers to the file actually read
    BOOST_CHECK_EQUAL(header.getInfo<std::string>("forwarding-generator_inputFile", isvalid), noHeader);
    BOOST_CHECK_EQUAL(header.getInfo<int>("forwarding-generator_inputEventNumber", isvalid), ev);
  }
}

/// a file which fails to open must not destroy the file that is currently open:
/// with roundRobin the open file is restarted in place, without being reopened
BOOST_AUTO_TEST_CASE(Rollover_FailedOpenKeepsCurrentFile)
{
  TempDir tmpDirGuard("rollover_keepfile");
  auto const& tmpDir = tmpDirGuard.path;
  auto good = (tmpDir / "good.root").string();
  createKineFile(good, 3, 0);
  auto nonexisting = (tmpDir / "doesnotexist.root").string();

  o2::eventgen::O2KineGenConfig config;
  config.roundRobin = true;
  o2::eventgen::GeneratorFromO2Kine gen(config, {good, nonexisting});
  auto filesOpenBefore = openRootFiles();
  BOOST_CHECK(gen.Init());

  // three passes over the only usable file
  for (int pass = 0; pass < 3; ++pass) {
    for (int ev = 0; ev < 3; ++ev) {
      BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(0, ev), 1E-6);
      BOOST_CHECK_EQUAL(openRootFiles() - filesOpenBefore, 1);
    }
  }
  // the good file has been opened exactly once, it was never closed and reopened
  BOOST_CHECK_EQUAL(gen.getNumberOfFilesUsed(), 1);
  BOOST_CHECK_EQUAL(gen.getCurrentFileIndex(), 0);
}

/// the wrap-around to the beginning of the list is reported also when it happens
/// after skipping an unusable file at the end of the list
BOOST_AUTO_TEST_CASE(Rollover_WrapReportedAfterSkippedFile)
{
  TempDir tmpDirGuard("rollover_wrapmessage");
  auto const& tmpDir = tmpDirGuard.path;
  auto good0 = (tmpDir / "good0.root").string();
  auto good1 = (tmpDir / "good1.root").string();
  createKineFile(good0, 2, 0);
  createKineFile(good1, 2, 1);
  auto nonexisting = (tmpDir / "doesnotexist.root").string();

  o2::eventgen::O2KineGenConfig config;
  config.roundRobin = true;
  o2::eventgen::GeneratorFromO2Kine gen(config, {good0, good1, nonexisting});
  LogCatcher log("rollover_wrapmessage", fair::Severity::info);
  BOOST_CHECK(gen.Init());
  for (int i = 0; i < 4; ++i) {
    readNextEvent(gen);
  }
  BOOST_CHECK_EQUAL(log.count("Reached the end of the input file list"), 0);
  // the last file is skipped, then the search wraps to the first one
  BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(0, 0), 1E-6);
  BOOST_CHECK_EQUAL(log.count("Reached the end of the input file list"), 1);
}

/// the start event refers to an entry of the first file: it is applied in sequential
/// order only, with randomize it is ignored (with a warning) and every event is served
BOOST_AUTO_TEST_CASE(StartEvent_OnlyWithoutRandomize)
{
  TempDir tmpDirGuard("start_event");
  auto file = (tmpDirGuard.path / "kine.root").string();
  constexpr int nevents = 5;
  createKineFile(file, nevents, 0);
  {
    o2::eventgen::GeneratorFromO2Kine gen(std::vector<std::string>{file});
    gen.SetStartEvent(2);
    BOOST_CHECK(gen.Init());
    BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(0, 2), 1E-6);
    BOOST_CHECK_CLOSE(readNextEvent(gen), encodeMomentum(0, 3), 1E-6);
  }
  {
    LogCatcher log("start_event_random", fair::Severity::warn);
    o2::eventgen::O2KineGenConfig config;
    config.randomize = true;
    o2::eventgen::GeneratorFromO2Kine gen(config, {file});
    gen.SetStartEvent(2);
    BOOST_CHECK(gen.Init());
    BOOST_CHECK_EQUAL(log.count("Start event 2 ignored"), 1);
    std::set<double> seen;
    for (int ev = 0; ev < nevents; ++ev) {
      seen.insert(readNextEvent(gen));
    }
    BOOST_CHECK_EQUAL(seen.size(), static_cast<size_t>(nevents));
  }
}

/// the number of events a generator expects to serve defaults to the one of the job,
/// but can be set (or declared unknown with 0), as done for sub-generators of a hybrid
BOOST_AUTO_TEST_CASE(ExpectedNEvents)
{
  unsigned int total = 42;
  o2::eventgen::Generator::setTotalNEvents(total);
  TempDir tmpDirGuard("expected_nevents");
  auto filenames = createPool(tmpDirGuard.path, 1);

  o2::eventgen::GeneratorFromO2Kine gen(filenames);
  BOOST_CHECK_EQUAL(gen.getExpectedNEvents(), 42u);
  gen.setExpectedNEvents(0);
  BOOST_CHECK_EQUAL(gen.getExpectedNEvents(), 0u);
  gen.setExpectedNEvents(7);
  BOOST_CHECK_EQUAL(gen.getExpectedNEvents(), 7u);

  // the internal generator of the event pool follows the one of the pool
  o2::eventgen::EventPoolGenConfig poolConfig;
  poolConfig.eventPoolPath = tmpDirGuard.path.string();
  o2::eventgen::GeneratorFromEventPool pool(poolConfig);
  pool.setExpectedNEvents(0);
  BOOST_CHECK(pool.Init());
  BOOST_CHECK_EQUAL(pool.getO2KineGenerator()->getExpectedNEvents(), 0u);

  total = 0;
  o2::eventgen::Generator::setTotalNEvents(total);
}

/// a single file, read without round robin, must NOT be silently reopened/reused once
/// exhausted: openNextFile()'s "next == mCurrentFileIndex && mCurrentFile" shortcut
/// (which restarts the currently open file in place) may only ever fire when wrapAround
/// (i.e. roundRobin) is true; with roundRobin off, every event is served exactly once and
/// asking for one more is a fatal condition
BOOST_AUTO_TEST_CASE(SingleFile_NoRoundRobin_ExhaustionIsFatal)
{
  LogCatcher log("exhaustion_single", fair::Severity::fatal);
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
  BOOST_CHECK_EQUAL(log.count("ran out of events"), 0);

  // the next request is fatal, and so is every further one: the file is not silently restarted
  BOOST_CHECK_THROW(readNextEvent(gen), fair::FatalException);
  BOOST_CHECK_EQUAL(log.count("ran out of events after 3 event(s) from 1 input file(s)"), 1);
  BOOST_CHECK_THROW(readNextEvent(gen), fair::FatalException);
  BOOST_CHECK_EQUAL(gen.getEventsServed(), nevents);
  BOOST_CHECK_EQUAL(gen.getNumberOfFilesUsed(), 1);
}

/// the pool of several files is fatal only after its LAST file is used up, and the message
/// reports what was served and (if known) what was requested
BOOST_AUTO_TEST_CASE(MultipleFiles_NoRoundRobin_ExhaustionIsFatal)
{
  LogCatcher log("exhaustion_multi", fair::Severity::fatal);
  TempDir tmpDirGuard("multi_file_no_rr");
  constexpr int numfiles = 3;
  auto filenames = createPool(tmpDirGuard.path, numfiles);
  int total = 0;
  for (int i = 0; i < numfiles; ++i) {
    total += eventsInFile(i);
  }

  o2::eventgen::GeneratorFromO2Kine gen(filenames);
  gen.setExpectedNEvents(total + 3);
  BOOST_CHECK(gen.Init());
  for (int i = 0; i < total; ++i) {
    BOOST_CHECK(readNextEvent(gen) >= 0.);
  }
  BOOST_CHECK_EQUAL(log.count("ran out of events"), 0);
  BOOST_CHECK_THROW(readNextEvent(gen), fair::FatalException);
  BOOST_CHECK_EQUAL(log.count("ran out of events after " + std::to_string(total) + " event(s) from 3 input file(s) (" + std::to_string(total + 3) + " were requested)"), 1);
  BOOST_CHECK_EQUAL(gen.getEventsServed(), total);
  BOOST_CHECK_EQUAL(gen.getNumberOfFilesUsed(), numfiles);

  // an unknown number of requested events is not quoted
  LogCatcher log2("exhaustion_multi_unknown", fair::Severity::fatal);
  o2::eventgen::GeneratorFromO2Kine gen2(filenames);
  gen2.setExpectedNEvents(0);
  BOOST_CHECK(gen2.Init());
  for (int i = 0; i < total; ++i) {
    BOOST_CHECK(readNextEvent(gen2) >= 0.);
  }
  BOOST_CHECK_THROW(readNextEvent(gen2), fair::FatalException);
  BOOST_CHECK_EQUAL(log2.count("ran out of events after " + std::to_string(total) + " event(s) from 3 input file(s)."), 1);
  BOOST_CHECK_EQUAL(log2.count("were requested"), 0);
}

/// the same for the event pool generator, which is what -g evtpool / the hybrid generator use
BOOST_AUTO_TEST_CASE(EventPool_NoRoundRobin_ExhaustionIsFatal)
{
  LogCatcher log("exhaustion_pool", fair::Severity::fatal);
  TempDir tmpDirGuard("pool_no_rr");
  constexpr int numfiles = 3;
  createPool(tmpDirGuard.path, numfiles);
  int total = 0;
  for (int i = 0; i < numfiles; ++i) {
    total += eventsInFile(i);
  }

  o2::eventgen::EventPoolGenConfig poolConfig;
  poolConfig.eventPoolPath = tmpDirGuard.path.string();
  poolConfig.roundRobin = false;
  poolConfig.randomize = true;
  poolConfig.rngseed = 17;
  o2::eventgen::GeneratorFromEventPool pool(poolConfig);
  BOOST_REQUIRE(pool.Init());

  // every event of the pool is served exactly once ...
  std::set<double> seen;
  for (int i = 0; i < total; ++i) {
    auto id = readNextEvent(pool);
    BOOST_CHECK(id >= 0.);
    BOOST_CHECK(seen.insert(id).second);
  }
  BOOST_CHECK_EQUAL(seen.size(), static_cast<size_t>(total));
  // ... and asking for one more is fatal
  BOOST_CHECK_THROW(readNextEvent(pool), fair::FatalException);
  BOOST_CHECK_EQUAL(log.count("ran out of events after " + std::to_string(total) + " event(s) from 3 input file(s)"), 1);
}

/// a pool path which holds no pool file stops the job at initialisation (LOG(fatal)), as a missing input did
/// before the chaining was introduced; the generator would otherwise be used uninitialised and crash
BOOST_AUTO_TEST_CASE(EventPool_NoPoolFile_InitIsFatal)
{
  LogCatcher log("pool_nofile", fair::Severity::fatal);
  TempDir tmpDirGuard("pool_nofile");
  o2::eventgen::EventPoolGenConfig poolConfig;
  poolConfig.eventPoolPath = (tmpDirGuard.path / "does_not_exist").string();
  o2::eventgen::GeneratorFromEventPool pool(poolConfig);
  BOOST_CHECK_THROW(pool.Init(), fair::FatalException);
  BOOST_CHECK_EQUAL(log.count("No file found that can be used with EventPool generator"), 1);

  // an existing directory without any pool file in it
  o2::eventgen::EventPoolGenConfig emptyConfig;
  emptyConfig.eventPoolPath = tmpDirGuard.path.string();
  o2::eventgen::GeneratorFromEventPool emptyPool(emptyConfig);
  BOOST_CHECK_THROW(emptyPool.Init(), fair::FatalException);
  BOOST_CHECK_EQUAL(log.count("No file found that can be used with EventPool generator"), 2);
}

/// with roundRobin the very same request is NOT fatal: the pool is started over
BOOST_AUTO_TEST_CASE(EventPool_RoundRobin_ExhaustionIsNotFatal)
{
  LogCatcher log("exhaustion_pool_rr", fair::Severity::warn);
  TempDir tmpDirGuard("pool_rr");
  constexpr int numfiles = 2;
  createPool(tmpDirGuard.path, numfiles);
  int total = 0;
  for (int i = 0; i < numfiles; ++i) {
    total += eventsInFile(i);
  }

  o2::eventgen::EventPoolGenConfig poolConfig;
  poolConfig.eventPoolPath = tmpDirGuard.path.string();
  poolConfig.roundRobin = true;
  poolConfig.randomize = false;
  poolConfig.rngseed = 17;
  o2::eventgen::GeneratorFromEventPool pool(poolConfig);
  BOOST_REQUIRE(pool.Init());
  for (int i = 0; i < 3 * total; ++i) {
    BOOST_CHECK_NO_THROW(BOOST_CHECK(readNextEvent(pool) >= 0.));
  }
  BOOST_CHECK_EQUAL(log.count("ran out of events"), 0);
}

/// the "input too small" estimate at initialisation uses the expected number of events of
/// the generator: the one of the job by default, none when declared unknown (sub-generators
/// of a hybrid only serve a share of the job and must not complain about the whole of it)
BOOST_AUTO_TEST_CASE(Init_WarnsOnlyForKnownExpectation)
{
  TempDir tmpDirGuard("init_warning");
  auto filenames = createPool(tmpDirGuard.path, 1); // 2 events in total
  unsigned int total = 10;
  o2::eventgen::Generator::setTotalNEvents(total);
  {
    LogCatcher log("init_warning_default", fair::Severity::warn);
    o2::eventgen::GeneratorFromO2Kine gen(filenames);
    BOOST_CHECK(gen.Init());
    BOOST_CHECK_EQUAL(log.count("holds only about 2"), 1);
  }
  {
    LogCatcher log("init_warning_unknown", fair::Severity::warn);
    o2::eventgen::GeneratorFromO2Kine gen(filenames);
    gen.setExpectedNEvents(0);
    BOOST_CHECK(gen.Init());
    BOOST_CHECK_EQUAL(log.count("holds only about"), 0);
  }
  {
    // an expectation which fits in the pool does not warn either, whatever the job asks for
    LogCatcher log("init_warning_fits", fair::Severity::warn);
    o2::eventgen::GeneratorFromO2Kine gen(filenames);
    gen.setExpectedNEvents(2);
    BOOST_CHECK(gen.Init());
    BOOST_CHECK_EQUAL(log.count("holds only about"), 0);
  }
  total = 0;
  o2::eventgen::Generator::setTotalNEvents(total);
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

/// the pool can be given as a text file listing the pool files (one per line): all the listed files, and
/// only those, are chained
BOOST_AUTO_TEST_CASE(EventPool_FileListChainsListedFiles)
{
  TempDir tmpDirGuard("evtpool_filelist");
  auto const& tmpDir = tmpDirGuard.path;
  auto files = createPool(tmpDir, 4);
  // leave out the file with index 1
  const std::vector<int> listed = {0, 2, 3};
  auto listPath = tmpDir / "pools.txt";
  {
    std::ofstream list(listPath);
    for (auto i : listed) {
      list << files[i] << "\n";
    }
  }
  int expectedEvents = 0;
  std::set<double> expectedIds;
  for (auto i : listed) {
    expectedEvents += eventsInFile(i);
    for (int ev = 0; ev < eventsInFile(i); ++ev) {
      expectedIds.insert(encodeMomentum(i, ev));
    }
  }

  o2::eventgen::EventPoolGenConfig config;
  config.eventPoolPath = listPath.string();
  config.randomize = false;
  config.rngseed = 7;
  o2::eventgen::GeneratorFromEventPool gen(config);
  BOOST_REQUIRE(gen.Init());
  BOOST_CHECK_EQUAL(gen.getFileUniverse().size(), listed.size());
  BOOST_CHECK_EQUAL(gen.getO2KineGenerator()->getNumberOfFiles(), static_cast<int>(listed.size()));

  std::set<double> seen;
  for (int i = 0; i < expectedEvents; ++i) {
    BOOST_CHECK(seen.insert(readNextEvent(gen)).second); // no duplicates
  }
  BOOST_CHECK(seen == expectedIds);
  BOOST_CHECK_EQUAL(gen.getO2KineGenerator()->getNumberOfFilesUsed(), static_cast<int>(listed.size()));
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

/// for a given seed the order must not depend on the order in which the files are
/// found (a directory listing or alien find gives no fixed order)
BOOST_AUTO_TEST_CASE(EventPool_SelectionIndependentOfListingOrder)
{
  TempDir tmpDirGuard("evtpool_listorder");
  auto const& tmpDir = tmpDirGuard.path;
  auto files = createPool(tmpDir, 8);

  auto orderFor = [&tmpDir](std::vector<std::string> const& listing, std::string const& name) {
    auto listPath = tmpDir / name;
    {
      std::ofstream list(listPath);
      for (auto const& f : listing) {
        list << f << "\n";
      }
    }
    o2::eventgen::EventPoolGenConfig config;
    config.eventPoolPath = listPath.string();
    config.rngseed = 3;
    o2::eventgen::GeneratorFromEventPool gen(config);
    gen.Init();
    return gen.getChosenFiles();
  };

  auto reversed = files;
  std::reverse(reversed.begin(), reversed.end());
  auto a = orderFor(files, "pools_a.txt");
  auto b = orderFor(reversed, "pools_b.txt");
  BOOST_CHECK_EQUAL(a.size(), files.size());
  BOOST_CHECK(a == b);
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
