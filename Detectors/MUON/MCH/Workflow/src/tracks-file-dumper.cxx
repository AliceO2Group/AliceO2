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

#include "boost/program_options.hpp"
#include <iostream>
#include <fstream>
#include <fmt/format.h>
#include "TrackAtVtxStruct.h"
#include "MCHBase/TrackBlock.h"
#include <gsl/span>
#include "DataFormatsMCH/TrackMCH.h"
#include "MCHBase/ClusterBlock.h"

namespace po = boost::program_options;

using o2::mch::ClusterStruct;
using o2::mch::TrackAtVtxStruct;
using o2::mch::TrackMCH;

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
    std::vector<ClusterStruct> clusters;
    // read the tracks, tracks at vertex and clusters (mind the reverse order of tracks
    // compared to the numbers above)
    if (!readBinaryStruct<TrackAtVtxStruct>(in, nofTracksAtVertex, tracksAtVertex, "TracksAtVertex")) {
      return -1;
    }
    if (!readBinaryStruct<TrackMCH>(in, nofTracks, tracks, "Tracks")) {
      return -1;
    }
    if (!readBinaryStruct<ClusterStruct>(in, nofAttachedClusters, clusters, "AttachedClusters")) {
      return -1;
    }

    dump(std::cout, tracksAtVertex);
  }
  return 0;
}

//     for (const auto& rof : rofs) {
//
//       // get the MCH tracks, attached clusters and corresponding tracks at vertex (if any)
//       auto eventClusters = getEventTracksAndClusters(rof, tracks, clusters, eventTracks);
//       auto eventTracksAtVtx = getEventTracksAtVtx(tracksAtVtx, tracksAtVtxOffset);
//
//       // write the number of tracks at vertex, MCH tracks and attached clusters
//       int nEventTracksAtVtx = eventTracksAtVtx.size() / sizeof(TrackAtVtxStruct);
//       mOutputFile.write(reinterpret_cast<char*>(&nEventTracksAtVtx), sizeof(int));
//       int nEventTracks = eventTracks.size();
//       mOutputFile.write(reinterpret_cast<char*>(&nEventTracks), sizeof(int));
//       int nEventClusters = eventClusters.size();
//       mOutputFile.write(reinterpret_cast<char*>(&nEventClusters), sizeof(int));
//
//       // write the tracks at vertex, MCH tracks and attached clusters
//       mOutputFile.write(eventTracksAtVtx.data(), eventTracksAtVtx.size());
//       mOutputFile.write(reinterpret_cast<const char*>(eventTracks.data()), eventTracks.size() * sizeof(TrackMCH));
//       mOutputFile.write(reinterpret_cast<const char*>(eventClusters.data()), eventClusters.size_bytes());
//     }
