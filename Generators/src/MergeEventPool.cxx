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
/// Unlike a naive TFileMerger-style concatenation (as done by O2DPG's root_merger.py),
/// this tool renumbers MCEventHeader's event ID so that it stays unique and monotonically
/// increasing across the merged output, instead of resetting per input file. No other
/// remapping is needed: each tree entry already holds one full, self-contained event, so
/// MCTrack mother/daughter indices (which are local to that entry) remain valid as-is.

#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/TrackReference.h"
#include <fairlogger/Logger.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <boost/program_options.hpp>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace bpo = boost::program_options;
namespace fs = std::filesystem;

namespace
{
const char* kTrackBranch = "MCTrack";
const char* kHeaderBranch = "MCEventHeader.";
const char* kTrackRefBranch = "TrackRefs";

// Checks that every input file exists, is readable, and has a tree with the branches
// this tool needs to merge (MCTrack + MCEventHeader are mandatory, TrackRefs optional
// but must be consistently present or absent across all files). Nothing is written
// until all inputs pass this check.
bool checkFiles(std::vector<std::string> const& files, std::string const& treename, bool& hasTrackRefs)
{
  bool ok = true;
  bool first = true;
  for (auto const& f : files) {
    if (!fs::exists(f)) {
      LOG(error) << "Input file " << f << " does not exist";
      ok = false;
      continue;
    }
    std::unique_ptr<TFile> file(TFile::Open(f.c_str(), "READ"));
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
    if (!hasTracks || !hasHeader) {
      LOG(error) << "File " << f << " is missing the required '" << kTrackBranch << "' and/or '" << kHeaderBranch << "' branch";
      ok = false;
      continue;
    }
    if (first) {
      hasTrackRefs = hasRefs;
      first = false;
    } else if (hasTrackRefs != hasRefs) {
      LOG(error) << "Inconsistent schema: file " << f << (hasRefs ? " has" : " lacks") << " a '" << kTrackRefBranch << "' branch, unlike previous input files";
      ok = false;
    }
    LOG(info) << "  OK  " << f << " (" << tree->GetEntries() << " events)";
  }
  return ok;
}

// Merges the already-validated input files into outfile, giving every event a fresh,
// globally unique, monotonically increasing event ID (starting at startId).
Long64_t mergeFiles(std::vector<std::string> const& files, std::string const& treename,
                     std::string const& outfile, bool hasTrackRefs, UInt_t startId)
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
  if (hasTrackRefs) {
    outTree->Branch(kTrackRefBranch, &trackrefsPtr);
  }

  UInt_t nextEventId = startId;
  Long64_t totalEvents = 0;

  for (auto const& f : files) {
    std::unique_ptr<TFile> fin(TFile::Open(f.c_str(), "READ"));
    auto tin = (TTree*)fin->Get(treename.c_str());

    tin->SetBranchAddress(kTrackBranch, &tracksPtr);
    tin->SetBranchAddress(kHeaderBranch, &headerPtr);
    if (hasTrackRefs) {
      tin->SetBranchAddress(kTrackRefBranch, &trackrefsPtr);
    }

    const Long64_t nEntries = tin->GetEntries();
    LOG(info) << "Merging " << nEntries << " events from " << f << " (event ID " << nextEventId << ".." << (nextEventId + nEntries - 1) << ")";
    for (Long64_t i = 0; i < nEntries; ++i) {
      tin->GetEntry(i);
      header->SetEventID(nextEventId++);
      outTree->Fill();
    }
    totalEvents += nEntries;
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
  options.add_options()
    ("input,i", bpo::value<std::string>()->required(), "comma-separated list of input event-pool ROOT files")
    ("output,o", bpo::value<std::string>()->required(), "output ROOT file with the merged event pool")
    ("treename,t", bpo::value<std::string>()->default_value("o2sim"), "name of the tree to merge")
    ("start-id", bpo::value<UInt_t>()->default_value(1), "event ID assigned to the first merged event")
    ("help,h", "produce help message");

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

  std::vector<std::string> infiles;
  {
    std::stringstream ss(vm["input"].as<std::string>());
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (!tok.empty()) {
        infiles.push_back(tok);
      }
    }
  }
  if (infiles.empty()) {
    LOG(fatal) << "No input files given";
    return 1;
  }

  const std::string outfile = vm["output"].as<std::string>();
  const std::string treename = vm["treename"].as<std::string>();
  const UInt_t startId = vm["start-id"].as<UInt_t>();

  LOG(info) << "Validating " << infiles.size() << " input file(s) ...";
  bool hasTrackRefs = false;
  if (!checkFiles(infiles, treename, hasTrackRefs)) {
    LOG(fatal) << "Validation failed; not writing any output";
    return 1;
  }

  LOG(info) << "Merging into " << outfile << " ...";
  const Long64_t total = mergeFiles(infiles, treename, outfile, hasTrackRefs, startId);
  if (total < 0) {
    return 1;
  }

  LOG(info) << "Done: wrote " << total << " events (event ID " << startId << ".." << (startId + total - 1) << ") to " << outfile;
  return 0;
}
