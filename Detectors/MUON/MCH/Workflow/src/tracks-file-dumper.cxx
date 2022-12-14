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

#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "MCHBase/TrackBlock.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TrackAtVtxStruct.h"
#include "TrackTreeReader.h"
#include "boost/program_options.hpp"
#include <TFile.h>
#include <TGrid.h>
#include <TTree.h>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <gsl/span>
#include <iostream>
#include <regex>

namespace po = boost::program_options;
namespace fs = std::filesystem;

using o2::mch::Cluster;
using o2::mch::Digit;
using o2::mch::ROFRecord;
using o2::mch::TrackAtVtxStruct;
using o2::mch::TrackMCH;
using o2::mch::TrackTreeReader;

template <typename T>
bool readBinaryStruct(std::istream& in, int nitems, std::vector<T>& items, const char* itemName)
{
  if (in.peek() == EOF) {
    return false;
  }
  // get the items if any
  if (nitems > 0) {
    auto offset = items.size();
    items.resize(offset + nitems);
    in.read(reinterpret_cast<char*>(&items[offset]), nitems * sizeof(T));
    if (in.fail()) {
      throw std::length_error(fmt::format("invalid input : cannot read {} {}", nitems, itemName));
    }
  }
  return true;
}

void dump(std::ostream& os, const o2::mch::TrackParamStruct& t)
{
  auto pt2 = t.px * t.px + t.py * t.py;
  auto p2 = t.pz * t.pz + pt2;
  auto pt = std::sqrt(pt2);
  auto p = std::sqrt(p2);
  os << fmt::format("({:s}) p {:7.2f} pt {:7.2f}", t.sign == -1 ? "-" : "+", p, pt);
}

void dump(std::ostream& os, gsl::span<o2::mch::TrackAtVtxStruct> tracksAtVertex)
{
  for (const auto& tv : tracksAtVertex) {
    os << fmt::format("id {:4d} ", tv.mchTrackIdx);
    dump(os, tv.paramAtVertex);
    os << fmt::format(" dca {:7.2f} rabs {:7.2f}", tv.dca, tv.rAbs)
       << "\n";
  }
}

void dump(std::ostream& os, const o2::mch::TrackMCH& t)
{
  auto pt = std::sqrt(t.getPx() * t.getPx() + t.getPy() * t.getPy());
  os << fmt::format("({:s}) p {:7.2f} pt {:7.2f} nclusters: {} \n", t.getSign() == -1 ? "-" : "+", t.getP(), pt, t.getNClusters());
}

int dumpBinary(std::string inputFile)
{
  std::ifstream in(inputFile.c_str());
  if (!in.is_open()) {
    std::cerr << "cannot open input file " << inputFile << "\n";
    return 3;
  }

  while (in.good()) {
    int nofTracksAtVertex{-1};
    int nofTracks{-1};
    int nofAttachedClusters{-1};
    // read the number of tracks at vertex, MCH tracks and attached clusters
    if (!in.read(reinterpret_cast<char*>(&nofTracksAtVertex), sizeof(int))) {
      return -1;
    }
    if (!in.read(reinterpret_cast<char*>(&nofTracks), sizeof(int))) {
      return -1;
    }
    if (!in.read(reinterpret_cast<char*>(&nofAttachedClusters), sizeof(int))) {
      return -1;
    }
    std::cout << fmt::format("=== nof MCH tracks: {:2d} at vertex: {:2d} w/ {:4d} attached clusters\n",
                             nofTracks, nofTracksAtVertex, nofAttachedClusters);
    std::vector<TrackAtVtxStruct> tracksAtVertex;
    std::vector<TrackMCH> tracks;
    std::vector<Cluster> clusters;
    // read the tracks, tracks at vertex and clusters (mind the reverse order of tracks
    // compared to the numbers above)
    if (!readBinaryStruct<TrackAtVtxStruct>(in, nofTracksAtVertex, tracksAtVertex, "TracksAtVertex")) {
      return -1;
    }
    if (!readBinaryStruct<TrackMCH>(in, nofTracks, tracks, "Tracks")) {
      return -1;
    }
    if (!readBinaryStruct<Cluster>(in, nofAttachedClusters, clusters, "AttachedClusters")) {
      return -1;
    }

    dump(std::cout, tracksAtVertex);
  }
  return 0;
}

int dumpRoot(std::string inputFile)
{
  if (std::regex_search(inputFile, std::regex("^alien://"))) {
    TGrid::Connect("alien");
  }
  std::unique_ptr<TFile> fin(TFile::Open(inputFile.c_str()));
  TTree* tree = static_cast<TTree*>(fin->Get("o2sim"));

  TrackTreeReader tr(tree);

  ROFRecord rof;
  std::vector<TrackMCH> tracks;
  std::vector<Cluster> clusters;
  std::vector<Digit> digits;
  std::vector<o2::MCCompLabel> labels;

  while (tr.next(rof, tracks, clusters, digits, labels)) {
    std::cout << rof << "\n";
    if (tr.hasLabels() && labels.size() != tracks.size()) {
      std::cerr << "the number of labels do not match the number of tracks\n";
      return -1;
    }
    int it(0);
    for (const auto& t : tracks) {
      std::cout << "   ";
      dump(std::cout, t);
      if (tr.hasLabels()) {
        std::cout << "   MC label: ";
        labels[it++].print();
      }
    }
  }
  return 0;
}

/**
 * o2-mch-tracks-file-dumper is a small helper program to inspect
 * track binary files (mch custom binary format for debug only)
 */

int main(int argc, char* argv[])
{
  std::string inputFile;
  po::variables_map vm;
  po::options_description options("options");

  // clang-format off
  // clang-format off
  options.add_options()
      ("help,h", "produce help message")
      ("infile,i", po::value<std::string>(&inputFile)->required(), "input file name")
      ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(options);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << options << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  std::string ext = fs::path(inputFile).extension();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });

  if (ext == ".root") {
    return dumpRoot(inputFile);
  }
  return dumpBinary(inputFile);
}
