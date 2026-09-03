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
/// written, and the merged pool is checked once more at the end. Anything that cannot be
/// resolved or opened aborts the merge, but this is bypassable with --skip-non-existing-files.
/// A failed run never leaves a file that looks finished (temporary filename during merge).
///
/// What went into the merge is written in the root file as a "mergeInfo" map
///
/// Usage:
///
///   # a few pools given directly
///   o2-generators-merge-evtpool -i poolA.root,poolB.root -o merged.root
///
///   # a local text file listing pools, which may be local and/or alien://
///   o2-generators-merge-evtpool -i pools.txt -o merged.root
///
/// Options: --input/-i (required), --output/-o (evtpool.root), --check-tree/-t (o2sim),
/// --skip-non-existing-files, --help/-h. Shell variables are expanded in every path, both in --input and inside
/// list files.
///
/// @author Marco Giacalone, mgiacalo@cern.ch, 08/2026

#include "CommonUtils/FileSystemUtils.h"
#include "CommonUtils/StringUtils.h"
#include <fairlogger/Logger.h>
#include <TFile.h>
#include <TFileMerger.h>
#include <TGrid.h>
#include <TMap.h>
#include <TObjString.h>
#include <TTree.h>
#include <boost/program_options.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
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
// Returns how many entries, here or in a nested list, could not be resolved.
size_t expandInputEntry(std::string const& rawEntry, std::vector<std::string>& out, std::vector<std::string>& stack)
{
  // done here so that the expansion works also when the variables appear in a list file
  const auto entry = o2::utils::expandShellVarsInFileName(rawEntry);
  if (o2::utils::Str::endsWith(entry, ".root")) {
    out.push_back(entry);
    return 0;
  }
  if (std::find(stack.begin(), stack.end(), entry) != stack.end()) {
    LOG(error) << "Reference to an existing list " << entry << "; ignoring";
    return 1;
  }
  auto lines = readLocalListFileLines(entry);
  if (!lines) {
    LOG(error) << "Cannot open " << entry << " (neither a .root file nor a readable local list)";
    return 1;
  }
  stack.push_back(entry);
  size_t unresolved = 0;
  for (auto line : *lines) {
    o2::utils::Str::trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    unresolved += expandInputEntry(line, out, stack);
  }
  stack.pop_back();
  return unresolved;
}

// Expands a list of raw --input entries (each either a .root file or a list) into the flat
// list of .root files to merge, dropping repetitions. Returns how many entries did not resolve.
size_t expandInputs(std::vector<std::string> const& rawEntries, std::vector<std::string>& infiles)
{
  std::vector<std::string> resolved;
  std::vector<std::string> stack;
  size_t unresolved = 0;
  for (auto const& e : rawEntries) {
    unresolved += expandInputEntry(e, resolved, stack);
  }
  std::set<std::string> seen;
  for (auto const& f : resolved) {
    if (seen.insert(f).second) {
      infiles.push_back(f);
    } else {
      LOG(warning) << "Input " << f << " is listed more than once; merging it only once";
    }
  }
  return unresolved;
}

// Checks that a file is readable and holds a tree with the branches expected from a
// standard o2-sim event pool, reporting its event count and compression settings.
// Returns an empty string when the file is usable, the reason otherwise.
std::string inspectFile(std::string const& path, std::string const& treename,
                        Long64_t& entries, int& compression)
{
  std::unique_ptr<TFile> file(TFile::Open(path.c_str(), "READ"));
  if (!file || file->IsZombie()) {
    return "file does not exist or cannot be opened";
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

// Checks every input before anything is written, collecting the usable ones and reporting
// the total number of events and the compression settings of the first usable input.
// Returns true when every input passed.
bool checkFiles(std::vector<std::string> const& files, std::string const& treename,
                std::vector<std::string>& usable, Long64_t& totalEvents, int& compression)
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
    usable.push_back(f);
    LOG(info) << "  OK  " << f << " (" << entries << " events)";
  }
  return ok;
}

// Records what the merge was asked for and what actually went into it.
void writeMergeInfo(std::string const& outfile, std::vector<std::string> const& requested,
                    std::vector<std::string> const& merged, size_t unresolved, Long64_t events)
{
  std::unique_ptr<TFile> file(TFile::Open(outfile.c_str(), "UPDATE"));
  if (!file || file->IsZombie()) {
    LOG(warning) << "Cannot add the merge information to " << outfile;
    return;
  }
  // the files that were asked for but did not make it, so that the gap can be named from
  // the file alone and not just counted
  std::string mergedList, skippedList;
  for (auto const& f : requested) {
    if (std::find(merged.begin(), merged.end(), f) != merged.end()) {
      mergedList += f + "\n";
    } else {
      skippedList += f + "\n";
    }
  }
  TMap info;
  info.SetOwnerKeyValue();
  info.Add(new TObjString("inputsRequested"), new TObjString(std::to_string(requested.size()).c_str()));
  info.Add(new TObjString("inputsMerged"), new TObjString(std::to_string(merged.size()).c_str()));
  info.Add(new TObjString("inputsUnresolved"), new TObjString(std::to_string(unresolved).c_str()));
  info.Add(new TObjString("events"), new TObjString(std::to_string(events).c_str()));
  info.Add(new TObjString("mergedFiles"), new TObjString(mergedList.c_str()));
  info.Add(new TObjString("skippedFiles"), new TObjString(skippedList.c_str()));
  file->cd();
  info.Write("mergeInfo", TObject::kSingleKey);
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
  add("check-tree,t", bpo::value<std::string>()->default_value("o2sim"),
      "name of the tree the inputs and the merged pool are checked against; everything the "
      "input files contain is merged regardless");
  add("skip-non-existing-files", bpo::bool_switch(),
      "skip inputs that cannot be resolved or opened instead of aborting the merge");
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
    LOG(error) << "Error parsing command-line arguments: " << e.what() << "\n\n"
               << options;
    return 1;
  }
  const auto rawEntries = o2::utils::Str::tokenize(vm["input"].as<std::string>(), ',');
  if (rawEntries.empty()) {
    LOG(error) << "No input files given";
    return 1;
  }
  // option similar in aodMerger
  const bool skipMissing = vm["skip-non-existing-files"].as<bool>();
  std::vector<std::string> infiles;
  const size_t unresolved = expandInputs(rawEntries, infiles);
  if (unresolved > 0 && !skipMissing) {
    LOG(error) << "Some --input entries could not be resolved; "
                  "pass --skip-non-existing-files to merge the rest anyway";
    return 1;
  }
  if (infiles.empty()) {
    LOG(error) << "No input files resolved from the given --input entries";
    return 1;
  }
  // Check Grid connection if any input is on AliEn
  if (std::any_of(infiles.begin(), infiles.end(), isAlienPath) && !GridOn()) {
    LOG(error) << "Some inputs live on AliEn but the grid is not available";
    return 1;
  }
  const std::string outfile = vm["output"].as<std::string>();
  const std::string treename = vm["check-tree"].as<std::string>();
  LOG(info) << "Validating " << infiles.size() << " input file(s) ...";
  std::vector<std::string> usable;
  Long64_t totalEvents = 0;
  int compression = -1;
  if (!checkFiles(infiles, treename, usable, totalEvents, compression) && !skipMissing) {
    LOG(error) << "Validation failed; not writing any output "
                  "(pass --skip-non-existing-files to merge the rest anyway)";
    return 1;
  }
  if (usable.empty()) {
    LOG(error) << "None of the input files could be used; not writing any output";
    return 1;
  }

  // merged into a temporary name and renamed only once the result has been checked, so that
  // a failed job never leaves something behind that looks like a finished pool
  const std::string partfile = outfile + ".part";
  auto discardPart = [&partfile]() {
    std::error_code ec;
    fs::remove(partfile, ec);
    return 1;
  };

  LOG(info) << "Merging " << totalEvents << " events from " << usable.size() << " file(s) into "
            << outfile << " ...";
  {
    TFileMerger merger(/*isLocal*/ false, /*histoOneGo*/ false);
    merger.SetPrintLevel(0);
    if (!merger.OutputFile(partfile.c_str(), "RECREATE", compression)) {
      LOG(error) << "Cannot create output file " << partfile;
      return discardPart();
    }
    for (auto const& f : usable) {
      if (!merger.AddFile(f.c_str())) {
        LOG(error) << "Cannot add " << f << " to the merge";
        return discardPart();
      }
    }
    if (!merger.Merge()) {
      LOG(error) << "Merging failed; no output written";
      return discardPart();
    }
  }

  writeMergeInfo(partfile, infiles, usable, unresolved, totalEvents);
  if (!validateOutput(partfile, treename, totalEvents)) {
    LOG(error) << "The merged pool did not pass the final check; no output written";
    return discardPart();
  }

  std::error_code ec;
  fs::rename(partfile, outfile, ec);
  if (ec) {
    LOG(error) << "Cannot move " << partfile << " to " << outfile << ": " << ec.message();
    return discardPart();
  }

  LOG(info) << "Done: wrote " << totalEvents << " events from " << usable.size() << " of "
            << infiles.size() << " input file(s) to " << outfile;
  return 0;
}
