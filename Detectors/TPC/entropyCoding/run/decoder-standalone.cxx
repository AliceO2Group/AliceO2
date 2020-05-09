// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   decoder-standalone.cxx
/// @author Michael Lettrich
/// @since  Apr 19, 2020
/// @brief standalone rans decoder for TPC compressed clusters
#include <cstring>

#include <TFile.h>
#include <boost/program_options.hpp>

#include "DataFormatsTPC/CompressedClusters.h"
#include "TPCEntropyCoding/EncodedClusters.h"
#include "TPCEntropyCoding/TPCEntropyDecoder.h"
#include "librans/rans.h"

namespace po = boost::program_options;

int main(int argc, char** argv)
{

  // specify command line interface:
  po::options_description description("o2-tpc rans-decoder-standalone Usage");
  // clang-format off
  description.add_options()
          ("help,h", "Display this help message")
          ("input,i", po::value<std::string>(), "File containing encoded TPC clusters")
          ("compare,c", po::value<std::string>(), "File containing uncompressed TPC clusters for cross check");
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
  const std::string comparisonFileName = vm.count("compare") ? vm["compare"].as<std::string>() : "";

  // read encoded clusters from ROOT file:
  std::unique_ptr<TFile>
    srcFile(TFile::Open(inputFileName.c_str()));
  assert(srcFile);
  std::unique_ptr<TTree> tree(reinterpret_cast<TTree*>(srcFile->Get("EncodedClusters")));
  assert(tree);

  auto encodedClusters = o2::tpc::TPCEntropyDecoder::fromTree(*tree);
  auto cc = o2::tpc::TPCEntropyDecoder::initCompressedClusters(*encodedClusters.get());

  // decode branches
  auto qTotA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->qTotA, "qTotA", *encodedClusters, cc->nAttachedClusters);
  auto qMaxA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->qMaxA, "qMaxA", *encodedClusters, cc->nAttachedClusters);
  auto flagsA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->flagsA, "flagsA", *encodedClusters, cc->nAttachedClusters);
  auto rowDiffA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->rowDiffA, "rowDiffA", *encodedClusters, cc->nAttachedClustersReduced);
  auto sliceLegDiffA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->sliceLegDiffA, "sliceLegDiffA", *encodedClusters, cc->nAttachedClustersReduced);
  auto padResA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->padResA, "padResA", *encodedClusters, cc->nAttachedClustersReduced);
  auto timeResA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->timeResA, "timeResA", *encodedClusters, cc->nAttachedClustersReduced);
  auto sigmaPadA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->sigmaPadA, "sigmaPadA", *encodedClusters, cc->nAttachedClusters);
  auto sigmaTimeA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->sigmaTimeA, "sigmaTimeA", *encodedClusters, cc->nAttachedClusters);
  auto qPtA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->qPtA, "qPtA", *encodedClusters, cc->nTracks);
  auto rowA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->rowA, "rowA", *encodedClusters, cc->nTracks);
  auto sliceA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->sliceA, "sliceA", *encodedClusters, cc->nTracks);
  auto timeA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->timeA, "timeA", *encodedClusters, cc->nTracks);
  auto padA = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->padA, "padA", *encodedClusters, cc->nTracks);
  auto qTotU = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->qTotU, "qTotU", *encodedClusters, cc->nUnattachedClusters);
  auto qMaxU = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->qMaxU, "qMaxU", *encodedClusters, cc->nUnattachedClusters);
  auto flagsU = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->flagsU, "flagsU", *encodedClusters, cc->nUnattachedClusters);
  auto padDiffU = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->padDiffU, "padDiffU", *encodedClusters, cc->nUnattachedClusters);
  auto timeDiffU = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->timeDiffU, "timeDiffU", *encodedClusters, cc->nUnattachedClusters);
  auto sigmaPadU = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->sigmaPadU, "sigmaPadU", *encodedClusters, cc->nUnattachedClusters);
  auto sigmaTimeU = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->sigmaTimeU, "sigmaTimeU", *encodedClusters, cc->nUnattachedClusters);
  auto nTrackClusters = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->nTrackClusters, "nTrackClusters", *encodedClusters, cc->nTracks);
  auto nSliceRowClusters = o2::tpc::TPCEntropyDecoder::decodeEntry(&cc->nSliceRowClusters, "nSliceRowClusters", *encodedClusters, cc->nSliceRows);

  if (!comparisonFileName.empty()) {
    // read compressed clusters from ROOT file:
    TFile file{comparisonFileName.c_str()};
    o2::tpc::CompressedClustersROOT* c;
    file.GetObject("TPCCompressedClusters", c);

    size_t numErrors = 0;
    auto checker = [&numErrors](int expr, std::string name) { if (expr != 0){LOG(ERROR) << name << " failed consistency check."; ++numErrors;} };

    checker(std::memcmp(cc->qTotA, c->qTotA, cc->nAttachedClusters * sizeof(unsigned short)), "qTotA");
    checker(std::memcmp(cc->qMaxA, c->qMaxA, cc->nAttachedClusters * sizeof(unsigned short)), "qMaxA");
    checker(std::memcmp(cc->flagsA, c->flagsA, cc->nAttachedClusters * sizeof(unsigned char)), "flagsA");
    checker(std::memcmp(cc->rowDiffA, c->rowDiffA, cc->nAttachedClustersReduced * sizeof(unsigned char)), "rowDiffA");
    checker(std::memcmp(cc->sliceLegDiffA, c->sliceLegDiffA, cc->nAttachedClustersReduced * sizeof(unsigned char)), "sliceLegDiffA");
    checker(std::memcmp(cc->padResA, c->padResA, cc->nAttachedClustersReduced * sizeof(unsigned short)), "padResA");
    checker(std::memcmp(cc->timeResA, c->timeResA, cc->nAttachedClustersReduced * sizeof(unsigned int)), "timeResA");
    checker(std::memcmp(cc->sigmaPadA, c->sigmaPadA, cc->nAttachedClusters * sizeof(unsigned char)), "sigmaPadA");
    checker(std::memcmp(cc->sigmaTimeA, c->sigmaTimeA, cc->nAttachedClusters * sizeof(unsigned char)), "sigmaTimeA");
    checker(std::memcmp(cc->qPtA, c->qPtA, cc->nTracks * sizeof(unsigned char)), "qPtA");
    checker(std::memcmp(cc->rowA, c->rowA, cc->nTracks * sizeof(unsigned char)), "rowA");
    checker(std::memcmp(cc->sliceA, c->sliceA, cc->nTracks * sizeof(unsigned char)), "sliceA");
    checker(std::memcmp(cc->timeA, c->timeA, cc->nTracks * sizeof(unsigned int)), "timeA");
    checker(std::memcmp(cc->padA, c->padA, cc->nTracks * sizeof(unsigned short)), "padA");
    checker(std::memcmp(cc->qTotU, c->qTotU, cc->nUnattachedClusters * sizeof(unsigned short)), "qTotU");
    checker(std::memcmp(cc->qMaxU, c->qMaxU, cc->nUnattachedClusters * sizeof(unsigned short)), "qMaxU");
    checker(std::memcmp(cc->flagsU, c->flagsU, cc->nUnattachedClusters * sizeof(unsigned char)), "flagsU");
    checker(std::memcmp(cc->padDiffU, c->padDiffU, cc->nUnattachedClusters * sizeof(unsigned short)), "padDiffU");
    checker(std::memcmp(cc->timeDiffU, c->timeDiffU, cc->nUnattachedClusters * sizeof(unsigned int)), "timeDiffU");
    checker(std::memcmp(cc->sigmaPadU, c->sigmaPadU, cc->nUnattachedClusters * sizeof(unsigned char)), "sigmaPadU");
    checker(std::memcmp(cc->sigmaTimeU, c->sigmaTimeU, cc->nUnattachedClusters * sizeof(unsigned char)), "sigmaTimeU");
    checker(std::memcmp(cc->nTrackClusters, c->nTrackClusters, cc->nTracks * sizeof(unsigned short)), "nTrackClusters");
    checker(std::memcmp(cc->nSliceRowClusters, c->nSliceRowClusters, cc->nSliceRows * sizeof(unsigned int)), "nSliceRowClusters");

    if (numErrors == 0) {
      LOG(INFO) << "encoded data passed consistency check";
    } else {
      LOG(ERROR) << "Data is inconsistent for " << numErrors << " arrays";
    }
  }
}
