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

/// \file EventPoolTestUtils.h
/// \brief helpers shared by the unit tests of the event pool chaining / hybrid generator
/// \author M. Giacalone, mgiacalo@cern.ch, 08/2026

#ifndef ALICEO2_EVENTGEN_TEST_EVENTPOOLTESTUTILS_H_
#define ALICEO2_EVENTGEN_TEST_EVENTPOOLTESTUTILS_H_

#include <fairlogger/Logger.h>
#include <Generators/Generator.h>
#include <Generators/GeneratorFromFile.h>
#include <SimulationDataFormat/MCEventHeader.h>
#include <SimulationDataFormat/MCTrack.h>

#include <TFile.h>
#include <TROOT.h>
#include <TTree.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <filesystem>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

namespace evtpooltest
{
namespace fs = std::filesystem;

/// the px of the first track of an event encodes the file it belongs to and its
/// position within that file; this allows to check the reading order later on
inline double encodeMomentum(int fileTag, int event)
{
  return 1000. * fileTag + event;
}

/// creates a minimal - but structurally valid - O2 kinematics file (optionally without
/// the MCEventHeader branch);
/// event `ev` contains `ev + 1` primary tracks (optionally linked as a chain)
inline void createKineFile(std::string const& path, int nevents, int fileTag, bool withHeader = true, bool chain = false)
{
  std::unique_ptr<TFile> file(TFile::Open(path.c_str(), "RECREATE"));
  auto tree = new TTree("o2sim", "o2sim"); // owned by the file
  std::vector<o2::MCTrack> tracks;
  tree->Branch("MCTrack", &tracks);
  o2::dataformats::MCEventHeader header;
  auto headerPtr = &header;
  if (withHeader) {
    tree->Branch("MCEventHeader.", &headerPtr);
  }

  for (int ev = 0; ev < nevents; ++ev) {
    tracks.clear();
    for (int i = 0; i <= ev; ++i) {
      // with `chain` the tracks of an event form a chain (track i is the mother of track i + 1), which
      // allows to check that the mother/daughter indices survive the merging done by a cocktail
      int mother = (chain && i > 0) ? i - 1 : -1;
      int daughter = (chain && i < ev) ? i + 1 : -1;
      o2::MCTrack track(211, mother, -1, daughter, daughter, encodeMomentum(fileTag, ev), 0., 0., 0., 0., 0., 0., 0);
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
inline std::vector<std::string> createPool(fs::path const& tmpDir, int nfiles, bool chain = false)
{
  std::vector<std::string> filenames;
  for (int i = 0; i < nfiles; ++i) {
    auto fileDir = tmpDir / std::to_string(i);
    fs::create_directories(fileDir);
    auto filePath = fileDir / o2::eventgen::GeneratorFromEventPool::eventpool_filename;
    createKineFile(filePath.string(), i + 2, i, /*withHeader=*/true, chain);
    filenames.push_back(filePath.string());
  }
  return filenames;
}

/// number of events in pool file i (must match createPool)
inline int eventsInFile(int i) { return i + 2; }

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
inline int openRootFiles()
{
  return gROOT->GetListOfFiles() ? gROOT->GetListOfFiles()->GetEntries() : 0;
}

/// reads the next event and returns the identifier encoded in the first track
inline double readNextEvent(o2::eventgen::Generator& gen)
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

/// Records the log messages of at least the given severity while it is alive, and makes LOG(fatal)
/// throw a fair::FatalException - which is what the O2 device runners do - instead of taking a
/// core dump. Without such a handler a fatal condition cannot be asserted from within a unit test.
/// The sink is called from whatever thread is logging, hence the mutex.
class LogCatcher
{
 public:
  explicit LogCatcher(std::string const& key, fair::Severity severity = fair::Severity::warn) : mKey(key)
  {
    fair::Logger::OnFatal([] { throw fair::FatalException("fatal condition (LOG(fatal)) raised in a unit test"); });
    fair::Logger::AddCustomSink(mKey, severity, [this](std::string const& content, fair::LogMetaData const&) {
      std::lock_guard<std::mutex> lock(mMutex);
      mMessages.push_back(content);
    });
  }
  ~LogCatcher() { fair::Logger::RemoveCustomSink(mKey); }
  LogCatcher(LogCatcher const&) = delete;
  LogCatcher& operator=(LogCatcher const&) = delete;

  /// number of recorded messages containing the given text
  int count(std::string const& text) const
  {
    std::lock_guard<std::mutex> lock(mMutex);
    return std::count_if(mMessages.begin(), mMessages.end(), [&](std::string const& m) { return m.find(text) != std::string::npos; });
  }
  /// the first recorded message containing the given text ("" if none)
  std::string find(std::string const& text) const
  {
    std::lock_guard<std::mutex> lock(mMutex);
    for (auto const& m : mMessages) {
      if (m.find(text) != std::string::npos) {
        return m;
      }
    }
    return "";
  }

 private:
  std::string mKey;
  mutable std::mutex mMutex;
  std::vector<std::string> mMessages;
};

} // namespace evtpooltest

#endif
