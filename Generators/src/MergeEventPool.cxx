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
/// This tool merges event pools with TFileMerger (the engine behind hadd).
///
/// Input handling is added in addition to hadd: files can be given
/// directly, or collected from text files listing further paths (one per line, '#'
/// comments allowed, resolved recursively). Both the pool files and the list files
/// themselves may live on AliEn (alien:// URLs) and are fetched via alien_cp (the same
/// idiom used for dynamic configuration files in GeneratorHybrid.cxx).
///
/// Pools are copied on purpose instead of merging them from SEs: a single bad replica
/// is enough to fail a read halfway through a long merge. A copy can be retried, while
/// a half-cloned output cannot. Inputs are handled in batches of --jobs files (downloaded
/// in parallel, checked, merged in one pass, deleted), so that the needed space on disk
/// stays at one batch. Transfers dominate a grid merge by orders of magnitude, so running
/// them in parallel is what makes merging many pools practical.
///
/// Inputs that cannot be fetched or merged are reported and skipped by default, so that a single bad
/// file does not throw away a long merge; --exit-on-failure stops at the first one
/// instead.
///
/// The final output is validated to hold the expected number of events and the required branches
///
/// Usage:
///
///   # a few pools given directly
///   o2-generators-merge-evtpool -i poolA.root,poolB.root -o merged.root
///
///   # a local text file listing pools (local and/or alien://), 8 transfers at a time
///   o2-generators-merge-evtpool -i pools.txt -o merged.root -j 8
///
///   # a list that itself lives on AliEn, combined with a local one
///   o2-generators-merge-evtpool -i pools.txt,alien:///alice/cern.ch/user/a/aliprod/pools.txt
///
/// Options: --input/-i (required), --output/-o (evtpool.root), --treename/-t (o2sim),
/// --tmpdir (/tmp), --jobs/-j (8, 0 = auto), --exit-on-failure, --help/-h.
/// Shell variables are expanded in every path, both in --input and inside list files.
///
/// @author Marco Giacalone, mgiacalo@cern.ch, 07-2026

#include "CommonUtils/FileSystemUtils.h"
#include "CommonUtils/StringUtils.h"
#include <fairlogger/Logger.h>
#include <TFile.h>
#include <TFileMerger.h>
#include <TTree.h>
#include <boost/program_options.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <unistd.h>
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
// number of attempts for AliEn fetches
constexpr int kFetchAttempts = 3;

bool isAlienPath(std::string const& path)
{
  return o2::utils::Str::beginsWith(path, kProtocol);
}

// Copies an AliEn file to dest, quietly. std::system() rather than TSystem::Exec() because
// this also runs from the worker threads that fetch a batch, and TSystem is not thread-safe.
bool alienCopy(std::string const& src, fs::path const& dest)
{
  const std::string cmd = "alien_cp " + src + " file:" + dest.string() + " > /dev/null 2>&1";
  return std::system(cmd.c_str()) == 0 && fs::exists(dest);
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
  static int counter = 0;
  const auto tmpPath = fs::temp_directory_path() /
                       fs::path("merge-evtpool-list-" + std::to_string(::getpid()) + "-" + std::to_string(counter++) + ".txt");

  std::optional<std::vector<std::string>> result;
  if (alienCopy(entry, tmpPath)) {
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
void expandInputEntry(std::string const& rawEntry, std::vector<std::string>& out, std::vector<std::string>& stack)
{
  // done here so that the expansion works also when the variables appear in a list file
  const auto entry = o2::utils::expandShellVarsInFileName(rawEntry);
  if (o2::utils::Str::endsWith(entry, ".root")) {
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
    o2::utils::Str::trim(line);
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

// Copies an AliEn file, retrying a few times
bool fetchFromAlien(std::string const& src, fs::path const& dest)
{
  for (int attempt = 0; attempt < kFetchAttempts; ++attempt) {
    if (alienCopy(src, dest)) {
      return true;
    }
    std::error_code ec;
    // remove, otherwise the partial copy can be mistaken for a good one
    fs::remove(dest, ec);
  }
  return false;
}

// Scratch directory for the local copies of the AliEn inputs, removed on destruction.
struct TempDir {
  fs::path path;
  explicit TempDir(fs::path const& base)
    : path(o2::utils::Str::create_unique_path((base / "merge-evtpool-").string()))
  {
    o2::utils::createDirectoriesIfAbsent(path.string());
  }
  ~TempDir()
  {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
  TempDir(TempDir const&) = delete;
  TempDir& operator=(TempDir const&) = delete;
};

// Definition of one input to be merged. `issue` is empty as long as the file is usable.
// local != source when the file was fetched from AliEn
struct Prepared {
  std::string source; // what the user asked for, used in messages
  std::string local;  // what the merger actually reads
  Long64_t events = 0;
  std::string issue;
};

// Checks that a (local) file is readable and holds a tree with the expected branches
// Returns an empty string when the file is usable, the issue otherwise.
std::string inspectFile(std::string const& path, std::string const& treename,
                        Long64_t& entries, int& compression)
{
  if (!fs::exists(path)) {
    return "file does not exist";
  }
  std::unique_ptr<TFile> file(TFile::Open(path.c_str(), "READ"));
  if (!file || file->IsZombie()) {
    return "file cannot be opened";
  }
  auto tree = (TTree*)file->Get(treename.c_str());
  if (!tree) {
    return "no tree named '" + treename + "' in the file";
  }
  if (tree->GetBranch(kTrackBranch) == nullptr || tree->GetBranch(kHeaderBranch) == nullptr ||
      tree->GetBranch(kTrackRefBranch) == nullptr) {
    return std::string("missing the required '") + kTrackBranch + "', '" + kHeaderBranch + "' and/or '" +
           kTrackRefBranch + "' branch";
  }
  entries = tree->GetEntries();
  compression = file->GetCompressionSettings();
  return {};
}

// Copies the AliEn inputs of one batch in parallel and checks every file of the batch.
// Anything that could not be fetched or does not look like an event pool is marked and left out of the merge.
void prepareBatch(std::vector<Prepared>& batch, std::string const& treename, fs::path const& tmpDir,
                  size_t firstIndex, int& compression)
{
  std::vector<std::thread> fetchers;
  for (size_t i = 0; i < batch.size(); ++i) {
    auto& p = batch[i];
    if (!isAlienPath(p.source)) {
      p.local = p.source;
      continue;
    }
    const auto dest = tmpDir / fs::path("input-" + std::to_string(firstIndex + i) + ".root");
    fetchers.emplace_back([&p, dest]() {
      if (fetchFromAlien(p.source, dest)) {
        p.local = dest.string();
      } else {
        p.issue = "could not be copied from AliEn after " + std::to_string(kFetchAttempts) + " attempts";
      }
    });
  }
  for (auto& t : fetchers) {
    t.join();
  }

  for (auto& p : batch) {
    if (!p.issue.empty()) {
      continue;
    }
    int fileCompression = -1;
    p.issue = inspectFile(p.local, treename, p.events, fileCompression);
    if (p.issue.empty() && compression < 0) {
      compression = fileCompression; // the first usable input sets the output compression
    }
  }
}

// Drops the local copies of a merged batch, while inputs that were already local are kept.
void dropCopies(std::vector<Prepared> const& batch)
{
  for (auto const& p : batch) {
    if (!p.local.empty() && p.local != p.source) {
      std::error_code ec;
      fs::remove(p.local, ec);
    }
  }
}

// Merges the usable files of one batch into outfile, appending to what is
// already there. No per-file retry is implemented: a failed partial merge could have written part of the batch
// so merging the same files again could duplicate events. For this reason the whole batch is dropped in case.
bool mergeBatch(std::vector<Prepared> const& batch, std::string const& outfile, bool& outputCreated,
                int compression)
{
  // usable file check before the output is opened: opening it with RECREATE would otherwise truncate
  // an already existing file
  const auto usable = std::count_if(batch.begin(), batch.end(),
                                    [](Prepared const& p) { return p.issue.empty(); });
  if (usable == 0) { // nothing survived the checks; leave the output as it was
    return true;
  }
  TFileMerger merger(/*isLocal*/ false, /*histoOneGo*/ false);
  merger.SetPrintLevel(0);
  if (!merger.OutputFile(outfile.c_str(), outputCreated ? "UPDATE" : "RECREATE", compression)) {
    LOG(error) << "Output file " << outfile << " cannot be written";
    return false;
  }
  for (auto const& p : batch) {
    if (p.issue.empty() && !merger.AddFile(p.local.c_str())) {
      LOG(error) << "Cannot add " << p.source << " to the merge";
      return false;
    }
  }
  if (!merger.PartialMerge(TFileMerger::kAll | TFileMerger::kIncremental)) {
    return false;
  }
  outputCreated = true;
  return true;
}

// Re-opens the merged output and checks if it contains the expected tree, branches and
// number of events. Very simple validation, that could be improved in the future
bool validateOutput(std::string const& outfile, std::string const& treename, Long64_t expected)
{
  Long64_t entries = 0;
  int compression = -1;
  const auto issue = inspectFile(outfile, treename, entries, compression);
  if (!issue.empty()) {
    LOG(error) << "Merged file " << outfile << " is not usable: " << issue;
    return false;
  }
  if (entries != expected) {
    LOG(error) << "Merged file " << outfile << " has " << entries << " events, but " << expected
               << " were merged into it";
    return false;
  }
  return true;
}
} // namespace

int main(int argc, char* argv[])
{
  bpo::options_description options("o2-generators-merge-evtpool options");
  auto add = options.add_options();
  add("input,i", bpo::value<std::string>()->required(),
      "comma-separated list of inputs: event-pool ROOT files, and/or text files (local or "
      "alien://, fetched via alien_cp) listing more paths (one per line, '#' comments allowed)");
  add("output,o", bpo::value<std::string>()->default_value("evtpool.root"),
      "output ROOT file with the merged event pool");
  add("treename,t", bpo::value<std::string>()->default_value("o2sim"), "name of the tree to merge");
  add("tmpdir", bpo::value<std::string>()->default_value("/tmp"),
      "directory for the local copies of AliEn inputs");
  add("jobs,j", bpo::value<unsigned int>()->default_value(8),
      "AliEn files fetched in parallel, which is also how many are merged per batch and how "
      "many copies exist at a time (0 = auto-detect)");
  add("exit-on-failure", bpo::bool_switch(),
      "stop at the first input that cannot be fetched or merged; by default such inputs are "
      "reported and skipped");
  add("help,h", "produce help message");

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

  const auto rawEntries = o2::utils::Str::tokenize(vm["input"].as<std::string>(), ',');
  if (rawEntries.empty()) {
    LOG(fatal) << "No input files given";
    return 1;
  }

  std::vector<std::string> infiles = expandInputs(rawEntries);
  if (infiles.empty()) {
    LOG(fatal) << "No input files resolved from the given --input entries";
    return 1;
  }

  const bool anyAlien = std::any_of(infiles.begin(), infiles.end(), isAlienPath);

  const std::string outfile = vm["output"].as<std::string>();
  const std::string treename = vm["treename"].as<std::string>();
  const bool exitOnFailure = vm["exit-on-failure"].as<bool>();
  size_t jobs = vm["jobs"].as<unsigned int>();
  if (jobs == 0) {
    jobs = std::max(1u, std::thread::hardware_concurrency());
  }

  // LOG(fatal) is avoided because it aborts the program, leaking the temporary copies
  TempDir tmpDir(vm["tmpdir"].as<std::string>());
  if (anyAlien && !fs::is_directory(tmpDir.path)) {
    LOG(error) << "Cannot create the temporary directory " << tmpDir.path;
    return 1;
  }

  // AliEn inputs are handled in batches: one batch is fetched in parallel, merged in one
  // pass and deleted, so temp space is limited at `batchSize` files.
  // Local inputs are simply merged in a single batch.
  const size_t batchSize = anyAlien ? jobs : infiles.size();
  LOG(info) << "Merging " << infiles.size() << " input file(s) into " << outfile
            << (anyAlien ? " (" + std::to_string(jobs) + " parallel transfers) ..." : " ...");

  Long64_t totalEvents = 0;
  int compression = -1;
  bool outputCreated = false;
  std::vector<std::string> skipped;
  for (size_t start = 0; start < infiles.size(); start += batchSize) {
    const size_t end = std::min(infiles.size(), start + batchSize);
    std::vector<Prepared> batch(end - start);
    for (size_t i = start; i < end; ++i) {
      batch[i - start].source = infiles[i];
    }

    prepareBatch(batch, treename, tmpDir.path, start, compression);
    if (!mergeBatch(batch, outfile, outputCreated, compression)) {
      for (auto& p : batch) {
        if (p.issue.empty()) {
          p.issue = "merging this batch into the output failed";
        }
      }
    }
    dropCopies(batch); // free the temp space before the next batch is fetched

    for (auto const& p : batch) {
      if (p.issue.empty()) {
        totalEvents += p.events;
        LOG(info) << "  merged  " << p.source << " (" << p.events << " events)";
        continue;
      }
      LOG(warning) << "Skipping " << p.source << ": " << p.issue;
      skipped.push_back(p.source);
      if (exitOnFailure) {
        LOG(error) << "Giving up because --exit-on-failure was requested";
        return 1;
      }
    }
  }

  if (!outputCreated) {
    LOG(error) << "None of the " << infiles.size() << " input file(s) could be merged; no output written";
    return 1;
  }

  if (!validateOutput(outfile, treename, totalEvents)) {
    LOG(error) << "The merged pool did not pass the final check";
    return 1;
  }

  LOG(info) << "Done: wrote " << totalEvents << " events from " << (infiles.size() - skipped.size())
            << " of " << infiles.size() << " input file(s) to " << outfile;
  if (!skipped.empty()) {
    LOG(warning) << skipped.size() << " input file(s) were skipped and are NOT part of " << outfile << ":";
    for (auto const& f : skipped) {
      LOG(warning) << "  " << f;
    }
  }
  return 0;
}
