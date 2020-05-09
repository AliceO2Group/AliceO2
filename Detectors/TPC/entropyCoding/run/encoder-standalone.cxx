// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   encoder-standalone.cxx
/// @author Michael Lettrich
/// @since  Apr 19, 2020
/// @brief standalone rans encoder for TPC compressed clusters
#include <iostream>
#include <string>

#include <TTree.h>
#include <TFile.h>
#include <Compression.h>

#include <boost/program_options.hpp>

#include "DataFormatsTPC/CompressedClusters.h"
#include "TPCEntropyCoding/EncodedClusters.h"
#include "TPCEntropyCoding/TPCEntropyEncoder.h"
#include "Framework/Logger.h"
#include "librans/rans.h"

namespace po = boost::program_options;

int main(int argc, char** argv)
{

  // specify command line interface:
  po::options_description description("o2-tpc rans-decoder-standalone Usage");
  // clang-format off
	  description.add_options()
	          ("help,h", "Display this help message")
	          ("input,i", po::value<std::string>(), "File containing encoded TPC clusters");
  // clang-format on

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(description).run(), vm);

  if (vm.count("help")) {
    std::cout << description;
    return 0;
  }
  if (!vm.count("input")) {
    std::cout << "No input specified, aborting" << std::endl;
    return 0;
  }

  po::notify(vm);

  //parse options;
  const std::string inputFileName = vm["input"].as<std::string>();

  // read compressed clusters from ROOT file:
  auto clusters = [&]() {
    TFile srcFile{inputFileName.c_str()};
    o2::tpc::CompressedClustersROOT* c;
    srcFile.GetObject("TPCCompressedClusters", c);
    srcFile.Close();
    if (c == nullptr) {
      throw std::runtime_error("Cannot read clusters from root file");
    }
    return std::unique_ptr<o2::tpc::CompressedClustersROOT>(c);
  }();

  std::cout << "nAttachedClusters: " << clusters->nAttachedClusters << std::endl;

  auto encodedClusters = o2::tpc::TPCEntropyEncoder::encode(*clusters);

  // create the tree
  TFile f("tpc-encoded-clusters.root", "recreate");
  TTree tree("EncodedClusters", "");
  o2::tpc::TPCEntropyEncoder::appendToTTree(tree, *encodedClusters);
}
