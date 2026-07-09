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

/// \brief Merges multiple event-pool files (e.g. evtpool.root / genevents_Kine.root,
///        produced by o2-sim --noGeant) into a single "o2sim" tree.
///
/// This tool merges event pools renumbering MCEventHeader's event ID so that it stays unique and
/// increasing across the merged output, instead of resetting per input file.
///
/// Input files can be given directly, or collected from text files listing further paths
/// (one per line, '#' comments allowed, resolved recursively). Both the pool files and the
/// list files themselves may live on AliEn (alien:// URLs); an AliEn list file is fetched
/// to a local temp copy via alien_cp (the same idiom used for dynamic configuration files
/// in GeneratorHybrid.cxx) and removed again once read. Reading of the input files is
/// parallelized: each file is deserialized by a worker thread into memory, while a single
/// writer thread fills the output tree strictly in input-file order, so the merged tree's
/// physical layout is unchanged with respect to a purely sequential merge.

#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/TrackReference.h"
#include <fairlogger/Logger.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TGrid.h>
#include <TROOT.h>
#include <TString.h>
#include <TSystem.h>
#include <boost/program_options.hpp>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <memory>
#include <optional>
#include <unistd.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace bpo = boost::program_options;
namespace fs = std::filesystem;

namespace
{
const char* kTrackBranch = "MCTrack";
const char* kHeaderBranch = "MCEventHeader.";
const char* kTrackRefBranch = "TrackRefs";
const char* kProtocol = "alien://";

bool isAlienPath(std::string const& path)
{
  return path.rfind(kProtocol, 0) == 0;
}

bool isRootFile(std::string const& path)
{
  std::string p = isAlienPath(path) ? path.substr(strlen(kProtocol)) : path;
  const std::string suffix = ".root";
  return p.size() >= suffix.size() && p.compare(p.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// TJAlienConnectionManager (behind TFile::Open("alien://...")) is not safe against
// concurrent Connect() calls: several worker threads opening AliEn files at once corrupt
// its shared websocket/TLS connection state and crash. The open/connect step is then serialized with a mutex
std::mutex gAlienOpenMutex;

std::unique_ptr<TFile> openInputFile(std::string const& path)
{
  if (isAlienPath(path)) {
    std::lock_guard<std::mutex> lk(gAlienOpenMutex);
    return std::unique_ptr<TFile>(TFile::Open(path.c_str(), "READ"));
  }
  return std::unique_ptr<TFile>(TFile::Open(path.c_str(), "READ"));
}

// Reads the lines of a local text file. nullopt if it could not be opened.
std::optional<std::vector<std::string>> readLocalListFileLines(std::string const& path)
{
  std::ifstream in(path);
  if (!in.is_open()) {
    return std::nullopt;
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  return lines;
}

// Fetches an AliEn list file to a local temp copy and reads its lines; the temp copy
// is removed before returning.
std::optional<std::vector<std::string>> readAlienListFileLines(std::string const& entry)
{
  if (!gGrid) {
    LOG(error) << "Not connected to AliEn; cannot read " << entry;
    return std::nullopt;
  }
  static std::atomic<int> counter{0};
  const auto tmpPath = fs::temp_directory_path() /
                       fs::path("merge-evtpool-list-" + std::to_string(::getpid()) + "-" + std::to_string(counter++) + ".txt");

  TString cmd = Form("alien_cp %s file:%s", entry.c_str(), tmpPath.c_str());
  const bool fetched = gSystem->Exec(cmd.Data()) == 0;

  std::optional<std::vector<std::string>> result;
  if (fetched) {
    result = readLocalListFileLines(tmpPath.string());
    if (!result) {
      LOG(error) << "Fetched " << entry << " from AliEn but could not read the local copy " << tmpPath.string();
    }
  } else {
    LOG(error) << "Failed to fetch list file " << entry << " from AliEn (alien_cp error)";
  }

  std::error_code ec;
  fs::remove(tmpPath, ec);
  return result;
}

// Returns the lines of a files list
std::optional<std::vector<std::string>> readListFileLines(std::string const& entry)
{
  return isAlienPath(entry) ? readAlienListFileLines(entry) : readLocalListFileLines(entry);
}

// Reads a text file listing input paths, one per line
// ('#' comments and blank lines ignored). Each listed path is either a .root file (local
// or alien://) or itself another list file.
void expandInputEntry(std::string const& entry, std::vector<std::string>& out, std::vector<std::string>& stack)
{
  if (isRootFile(entry)) {
    out.push_back(entry);
    return;
  }
  if (std::find(stack.begin(), stack.end(), entry) != stack.end()) {
    LOG(error) << "Reference to an existing list " << entry << "; ignoring";
    return;
  }
  auto lines = readListFileLines(entry);
  if (!lines) {
    LOG(error) << "Cannot open " << entry << " (neither a .root file nor a readable list)";
    return;
  }
  stack.push_back(entry);
  for (auto line : *lines) {
    const auto b = line.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
      continue;
    }
    const auto e = line.find_last_not_of(" \t\r\n");
    line = line.substr(b, e - b + 1);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    expandInputEntry(line, out, stack);
  }
  stack.pop_back();
}

// Expands a list of raw --input entries (each either a .root file or a list) into
// the flat list of .root files to merge.
std::vector<std::string> expandInputs(std::vector<std::string> const& rawEntries)
{
  std::vector<std::string> result;
  std::vector<std::string> stack;
  for (auto const& e : rawEntries) {
    expandInputEntry(e, result, stack);
  }
  return result;
}

// Checks that every input file exists, is readable, and has a tree with the branches
// this tool needs to merge, as produced by the standard o2-sim simulation.
bool checkFiles(std::vector<std::string> const& files, std::string const& treename, std::vector<Long64_t>& entryCounts)
{
  bool ok = true;
  entryCounts.assign(files.size(), 0);
  for (size_t i = 0; i < files.size(); ++i) {
    auto const& f = files[i];
    if (!isAlienPath(f) && !fs::exists(f)) {
      LOG(error) << "Input file " << f << " does not exist";
      ok = false;
      continue;
    }
    std::unique_ptr<TFile> file = openInputFile(f);
    if (!file || file->IsZombie()) {
      LOG(error) << "Cannot open " << f;
      ok = false;
      continue;
    }
    auto tree = (TTree*)file->Get(treename.c_str());
    if (!tree) {
      LOG(error) << "No tree named '" << treename << "' found in " << f;
      ok = false;
      continue;
    }
    const bool hasTracks = tree->GetBranch(kTrackBranch) != nullptr;
    const bool hasHeader = tree->GetBranch(kHeaderBranch) != nullptr;
    const bool hasRefs = tree->GetBranch(kTrackRefBranch) != nullptr;
    if (!hasTracks || !hasHeader || !hasRefs) {
      LOG(error) << "File " << f << " is missing the required '" << kTrackBranch << "', '" << kHeaderBranch
                 << "' and/or '" << kTrackRefBranch << "' branch";
      ok = false;
      continue;
    }
    entryCounts[i] = tree->GetEntries();
    LOG(info) << "  OK  " << f << " (" << entryCounts[i] << " events)";
  }
  return ok;
}

// One deserialized event, buffered in memory between the reader thread that produced it
// and the writer thread that fills the output tree.
struct Event {
  std::vector<o2::MCTrack> tracks;
  o2::dataformats::MCEventHeader header;
  std::vector<o2::TrackReference> trackrefs;
};

// Reads one whole input file into memory, assigning event IDs starting at startId.
std::vector<Event> readFile(std::string const& file, std::string const& treename, Long64_t nEntries, UInt_t startId)
{
  std::vector<Event> events;
  events.reserve(nEntries);

  std::unique_ptr<TFile> fin = openInputFile(file);
  if (!fin || fin->IsZombie()) {
    // the file passed validation earlier, so this only happens if it vanished in between
    LOG(fatal) << "File " << file << " became unreadable after validation";
  }
  auto tin = (TTree*)fin->Get(treename.c_str());

  auto tracks = std::make_unique<std::vector<o2::MCTrack>>();
  auto header = std::make_unique<o2::dataformats::MCEventHeader>();
  auto trackrefs = std::make_unique<std::vector<o2::TrackReference>>();
  auto* tracksPtr = tracks.get();
  auto* headerPtr = header.get();
  auto* trackrefsPtr = trackrefs.get();
  tin->SetBranchAddress(kTrackBranch, &tracksPtr);
  tin->SetBranchAddress(kHeaderBranch, &headerPtr);
  tin->SetBranchAddress(kTrackRefBranch, &trackrefsPtr);

  UInt_t id = startId;
  for (Long64_t i = 0; i < nEntries; ++i) {
    tin->GetEntry(i);
    headerPtr->SetEventID(id++);
    events.push_back(Event{*tracksPtr, *headerPtr, *trackrefsPtr});
  }
  return events;
}

// Merges the already-validated input files into outfile, giving every event a fresh,
// globally unique event ID (starting at startId).
//
// Reading is parallelized across up to `jobs` worker threads (one input file at a time
// per thread); a single writer (this thread) fills the output tree strictly in input-file
// order as soon as each file's buffer is ready.
Long64_t mergeFiles(std::vector<std::string> const& files, std::string const& treename,
                    std::string const& outfile, std::vector<Long64_t> const& entryCounts,
                    UInt_t startId, unsigned int jobs)
{
  TFile fout(outfile.c_str(), "RECREATE");
  if (fout.IsZombie()) {
    LOG(fatal) << "Cannot create output file " << outfile;
    return -1;
  }

  auto tracks = std::make_unique<std::vector<o2::MCTrack>>();
  auto header = std::make_unique<o2::dataformats::MCEventHeader>();
  auto trackrefs = std::make_unique<std::vector<o2::TrackReference>>();
  auto* tracksPtr = tracks.get();
  auto* headerPtr = header.get();
  auto* trackrefsPtr = trackrefs.get();

  auto outTree = new TTree(treename.c_str(), treename.c_str());
  outTree->Branch(kTrackBranch, &tracksPtr);
  outTree->Branch(kHeaderBranch, &headerPtr);
  outTree->Branch(kTrackRefBranch, &trackrefsPtr);

  // per-file starting event ID, computed upfront from the validated entry counts
  std::vector<UInt_t> startIds(files.size());
  {
    UInt_t next = startId;
    for (size_t i = 0; i < files.size(); ++i) {
      startIds[i] = next;
      next += static_cast<UInt_t>(entryCounts[i]);
    }
  }

  std::vector<std::vector<Event>> fileBuffers(files.size());
  std::vector<std::atomic<bool>> fileReady(files.size());
  for (auto& r : fileReady) {
    r.store(false);
  }
  std::atomic<size_t> nextFileToRead{0};
  std::atomic<size_t> nextFileToWrite{0};
  std::mutex readyMutex;
  std::condition_variable readyCv;

  const unsigned int nWorkers = std::max(1u, std::min<unsigned int>(jobs, files.size() > 0 ? files.size() : 1));
  // bound on how far readers may run ahead of the writer, so buffered-but-unwritten
  // files cannot pile up in memory when reading outpaces writing/compression
  const size_t maxFilesAhead = 2 * static_cast<size_t>(nWorkers);

  auto worker = [&]() {
    while (true) {
      const size_t idx = nextFileToRead.fetch_add(1);
      if (idx >= files.size()) {
        break;
      }
      {
        std::unique_lock<std::mutex> lk(readyMutex);
        readyCv.wait(lk, [&] { return idx < nextFileToWrite.load(std::memory_order_acquire) + maxFilesAhead; });
      }
      LOG(info) << "Reading " << entryCounts[idx] << " events from " << files[idx]
                << " (event ID " << startIds[idx] << ".." << (startIds[idx] + entryCounts[idx] - 1) << ")";
      fileBuffers[idx] = readFile(files[idx], treename, entryCounts[idx], startIds[idx]);
      {
        // the store must happen under the mutex: otherwise it can race with the writer's
        // check and the notification is lost (deadlock on the last file)
        std::lock_guard<std::mutex> lk(readyMutex);
        fileReady[idx].store(true, std::memory_order_release);
      }
      readyCv.notify_all();
    }
  };
  std::vector<std::thread> workers;
  workers.reserve(nWorkers);
  for (unsigned int w = 0; w < nWorkers; ++w) {
    workers.emplace_back(worker);
  }

  Long64_t totalEvents = 0;
  for (size_t idx = 0; idx < files.size(); ++idx) {
    {
      std::unique_lock<std::mutex> lk(readyMutex);
      readyCv.wait(lk, [&] { return fileReady[idx].load(std::memory_order_acquire); });
    }
    for (auto& ev : fileBuffers[idx]) {
      *tracksPtr = std::move(ev.tracks);
      *headerPtr = std::move(ev.header);
      *trackrefsPtr = std::move(ev.trackrefs);
      outTree->Fill();
    }
    totalEvents += static_cast<Long64_t>(fileBuffers[idx].size());
    // free the buffer as soon as it has been written out and let waiting readers advance
    fileBuffers[idx].clear();
    fileBuffers[idx].shrink_to_fit();
    {
      std::lock_guard<std::mutex> lk(readyMutex);
      nextFileToWrite.store(idx + 1, std::memory_order_release);
    }
    readyCv.notify_all();
  }

  for (auto& t : workers) {
    t.join();
  }

  fout.cd();
  outTree->Write("", TObject::kWriteDelete);
  fout.Close();

  return totalEvents;
}
} // namespace

int main(int argc, char* argv[])
{
  bpo::options_description options("o2-generators-merge-evtpool options");
  options.add_options()("input,i", bpo::value<std::string>()->required(),
                        "comma-separated list of inputs: event-pool ROOT files "
                        ", and/or text files (local or alien://, fetched via alien_cp) listing more "
                        "paths (one per line, '#' comments allowed)")("output,o", bpo::value<std::string>()->default_value("evtpool.root"), "output ROOT file with the merged event pool")("treename,t", bpo::value<std::string>()->default_value("o2sim"), "name of the tree to merge")("start-id", bpo::value<UInt_t>()->default_value(1), "event ID assigned to the first merged event")("jobs,j", bpo::value<unsigned int>()->default_value(8), "number of worker threads used to read input files in parallel (0 = auto-detect)")("help,h", "produce help message");

  bpo::variables_map vm;
  try {
    bpo::store(bpo::parse_command_line(argc, argv, options), vm);
    if (vm.count("help")) {
      LOG(info) << options;
      return 0;
    }
    bpo::notify(vm);
  } catch (const bpo::error& e) {
    LOG(fatal) << "Error parsing command-line arguments: " << e.what() << "\n\n"
               << options;
    return 1;
  }

  std::vector<std::string> rawEntries;
  {
    std::stringstream ss(vm["input"].as<std::string>());
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        rawEntries.push_back(tok);
      }
    }
  }
  if (rawEntries.empty()) {
    LOG(fatal) << "No input files given";
    return 1;
  }

  ROOT::EnableThreadSafety();

  const bool needsAlien = std::any_of(rawEntries.begin(), rawEntries.end(), isAlienPath);
  if (needsAlien && !gGrid) {
    LOG(info) << "Connecting to AliEn ...";
    if (!TGrid::Connect("alien:") || !gGrid) {
      LOG(fatal) << "Could not connect to AliEn; check your alien token";
      return 1;
    }
  }

  const std::vector<std::string> infiles = expandInputs(rawEntries);
  if (infiles.empty()) {
    LOG(fatal) << "No input files resolved from the given --input entries";
    return 1;
  }

  const std::string outfile = vm["output"].as<std::string>();
  const std::string treename = vm["treename"].as<std::string>();
  const UInt_t startId = vm["start-id"].as<UInt_t>();
  unsigned int jobs = vm["jobs"].as<unsigned int>();
  if (jobs == 0) {
    jobs = std::max(1u, std::thread::hardware_concurrency());
  }

  LOG(info) << "Validating " << infiles.size() << " input file(s) ...";
  std::vector<Long64_t> entryCounts;
  if (!checkFiles(infiles, treename, entryCounts)) {
    LOG(fatal) << "Validation failed; not writing any output";
    return 1;
  }

  LOG(info) << "Merging into " << outfile << " using up to " << jobs << " reader thread(s) ...";
  const Long64_t total = mergeFiles(infiles, treename, outfile, entryCounts, startId, jobs);
  if (total < 0) {
    return 1;
  }

  LOG(info) << "Done: wrote " << total << " events (event ID " << startId << ".." << (startId + total - 1) << ") to " << outfile;
  return 0;
}
