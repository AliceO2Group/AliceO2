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
/// Input handling is added in addition to hadd: files can be given directly, or collected
/// from local text files listing further paths (one per line, '#' comments allowed,
/// resolved recursively). The pools themselves can live on AliEn (alien:// URLs) and are
/// read straight from the storage elements
///
/// Every input is validated (tree and required branches present) before anything is
/// written, so a bad file is reported and nothing is produced, and the merged pool is
/// checked once more at the end.
///
/// Usage:
///
///   # a few pools given directly
///   o2-generators-merge-evtpool -i poolA.root,poolB.root -o merged.root
///
///   # a local text file listing pools, which may be local and/or alien://
///   o2-generators-merge-evtpool -i pools.txt -o merged.root
///
/// Options: --input/-i (required), --output/-o (evtpool.root), --treename/-t (o2sim),
/// --help/-h. Shell variables are expanded in every path, both in --input and inside
/// list files.
///
/// @author Marco Giacalone, mgiacalo@cern.ch, 08/2026

#include "CommonUtils/FileSystemUtils.h"
#include "CommonUtils/StringUtils.h"
#include <fairlogger/Logger.h>
#include <TFile.h>
#include <TFileMerger.h>
#include <TGrid.h>
#include <TTree.h>
#include <boost/program_options.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
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
  return o2::utils::Str::beginsWith(path, kProtocol);
}

// Connects to AliEn if that has not happened yet
bool GridOn()
{
  if (gGrid) {
    return true;
  }
  LOG(info) << "Connecting to AliEn ...";
  if (!TGrid::Connect("alien:") || !gGrid) {
    LOG(error) << "Could not connect to AliEn; check your alien token";
    return false;
  }
  return true;
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

// Reads a text file listing input paths, one per line ('#' comments and blank lines
// ignored). Each listed path is either a .root file (local or alien://) or itself
// another list file. The lists themselves are always read locally.
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
  auto lines = readLocalListFileLines(entry);
  if (!lines) {
    LOG(error) << "Cannot open " << entry << " (neither a .root file nor a readable local list)";
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

// Checks that a file is readable and holds a tree with the branches expected from a
// standard o2-sim event pool, reporting its event count and compression settings.
// Returns an empty string when the file is usable, the reason otherwise.
std::string inspectFile(std::string const& path, std::string const& treename,
                        Long64_t& entries, int& compression)
{
  if (!isAlienPath(path) && !fs::exists(path)) {
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

// Checks every input before anything is written. Reports the total number of events and
// the compression settings of the first input
bool checkFiles(std::vector<std::string> const& files, std::string const& treename,
                Long64_t& totalEvents, int& compression)
{
  bool ok = true;
  totalEvents = 0;
  compression = -1;
  for (auto const& f : files) {
    Long64_t entries = 0;
    int fileCompression = -1;
    const auto issue = inspectFile(f, treename, entries, fileCompression);
    if (!issue.empty()) {
      LOG(error) << "Input file " << f << ": " << issue;
      ok = false;
      continue;
    }
    if (compression < 0) {
      compression = fileCompression;
    }
    totalEvents += entries;
    LOG(info) << "  OK  " << f << " (" << entries << " events)";
  }
  return ok;
}

// Re-opens the merged output and checks that it holds the expected tree, branches and
// number of events, so that a truncated or half-written pool does not pass unnoticed.
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
      "comma-separated list of inputs: event-pool ROOT files (local or alien://), and/or "
      "local text files listing more paths (one per line, '#' comments allowed)");
  add("output,o", bpo::value<std::string>()->default_value("evtpool.root"),
      "output ROOT file with the merged event pool");
  add("treename,t", bpo::value<std::string>()->default_value("o2sim"), "name of the tree to merge");
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
  const auto infiles = expandInputs(rawEntries);
  if (infiles.empty()) {
    LOG(fatal) << "No input files resolved from the given --input entries";
    return 1;
  }
  // Check Grid connection if any input is on AliEn
  if (std::any_of(infiles.begin(), infiles.end(), isAlienPath) && !GridOn()) {
    LOG(fatal) << "Some inputs live on AliEn but the grid is not available";
    return 1;
  }
  const std::string outfile = vm["output"].as<std::string>();
  const std::string treename = vm["treename"].as<std::string>();
  LOG(info) << "Validating " << infiles.size() << " input file(s) ...";
  Long64_t totalEvents = 0;
  int compression = -1;
  if (!checkFiles(infiles, treename, totalEvents, compression)) {
    LOG(fatal) << "Validation failed; not writing any output";
    return 1;
  }
  LOG(info) << "Merging " << totalEvents << " events into " << outfile << " ...";
  TFileMerger merger(/*isLocal*/ false, /*histoOneGo*/ false);
  merger.SetPrintLevel(0);
  if (!merger.OutputFile(outfile.c_str(), "RECREATE", compression)) {
    LOG(fatal) << "Cannot create output file " << outfile;
    return 1;
  }
  for (auto const& f : infiles) {
    if (!merger.AddFile(f.c_str())) {
      LOG(fatal) << "Cannot add " << f << " to the merge";
      return 1;
    }
  }
  if (!merger.Merge()) {
    LOG(fatal) << "Merging failed; output " << outfile << " is incomplete";
    return 1;
  }
  if (!validateOutput(outfile, treename, totalEvents)) {
    LOG(fatal) << "The merged pool did not pass the final check";
    return 1;
  }
  LOG(info) << "Done: wrote " << totalEvents << " events to " << outfile;
  return 0;
}
