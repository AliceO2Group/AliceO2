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

/// @file   testClusterNative.cxx
/// @since  2018-01-17
/// @brief  Unit test for the TPC ClusterNative data struct

#define BOOST_TEST_MODULE Test TPC DataFormats
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include "../include/DataFormatsTPC/CompressedClusters.h"
#include "../include/DataFormatsTPC/CompressedClustersHelpers.h"
#include <boost/test/unit_test.hpp>
#include <string>
#include <vector>
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>

namespace o2::tpc
{

template <typename ContainerType>
void fillRandom(ContainerType& target, size_t size)
{
  target.resize(size);
  std::generate(target.begin(), target.end(), []() { return rand() % 256; });
}

struct ClustersData {
  std::vector<unsigned short> qTotA;        //! [nAttachedClusters]
  std::vector<unsigned short> qMaxA;        //! [nAttachedClusters]
  std::vector<unsigned char> flagsA;        //! [nAttachedClusters]
  std::vector<unsigned char> rowDiffA;      //! [nAttachedClustersReduced]
  std::vector<unsigned char> sliceLegDiffA; //! [nAttachedClustersReduced]
  std::vector<unsigned short> padResA;      //! [nAttachedClustersReduced]
  std::vector<unsigned int> timeResA;       //! [nAttachedClustersReduced]
  std::vector<unsigned char> sigmaPadA;     //! [nAttachedClusters]
  std::vector<unsigned char> sigmaTimeA;    //! [nAttachedClusters]

  std::vector<unsigned char> qPtA;   //! [nTracks]
  std::vector<unsigned char> rowA;   //! [nTracks]
  std::vector<unsigned char> sliceA; //! [nTracks]
  std::vector<unsigned int> timeA;   //! [nTracks]
  std::vector<unsigned short> padA;  //! [nTracks]

  std::vector<unsigned short> qTotU;     //! [nUnattachedClusters]
  std::vector<unsigned short> qMaxU;     //! [nUnattachedClusters]
  std::vector<unsigned char> flagsU;     //! [nUnattachedClusters]
  std::vector<unsigned short> padDiffU;  //! [nUnattachedClusters]
  std::vector<unsigned int> timeDiffU;   //! [nUnattachedClusters]
  std::vector<unsigned char> sigmaPadU;  //! [nUnattachedClusters]
  std::vector<unsigned char> sigmaTimeU; //! [nUnattachedClusters]

  std::vector<unsigned short> nTrackClusters;  //! [nTracks]
  std::vector<unsigned int> nSliceRowClusters; //! [nSliceRows]
};

void fillClusters(CompressedClustersROOT& clusters, ClustersData& data)
{
  clusters.nAttachedClusters = rand() % 32;
  fillRandom(data.qTotA, clusters.nAttachedClusters);
  clusters.qTotA = data.qTotA.data();
  fillRandom(data.qMaxA, clusters.nAttachedClusters);
  clusters.qMaxA = data.qMaxA.data();
  fillRandom(data.flagsA, clusters.nAttachedClusters);
  clusters.flagsA = data.flagsA.data();
  fillRandom(data.sigmaPadA, clusters.nAttachedClusters);
  clusters.sigmaPadA = data.sigmaPadA.data();
  fillRandom(data.sigmaTimeA, clusters.nAttachedClusters);
  clusters.sigmaTimeA = data.sigmaTimeA.data();

  clusters.nAttachedClustersReduced = rand() % 32;
  fillRandom(data.rowDiffA, clusters.nAttachedClustersReduced);
  clusters.rowDiffA = data.rowDiffA.data();
  fillRandom(data.sliceLegDiffA, clusters.nAttachedClustersReduced);
  clusters.sliceLegDiffA = data.sliceLegDiffA.data();
  fillRandom(data.padResA, clusters.nAttachedClustersReduced);
  clusters.padResA = data.padResA.data();
  fillRandom(data.timeResA, clusters.nAttachedClustersReduced);
  clusters.timeResA = data.timeResA.data();

  clusters.nTracks = rand() % 32;
  fillRandom(data.qPtA, clusters.nTracks);
  clusters.qPtA = data.qPtA.data();
  fillRandom(data.rowA, clusters.nTracks);
  clusters.rowA = data.rowA.data();
  fillRandom(data.sliceA, clusters.nTracks);
  clusters.sliceA = data.sliceA.data();
  fillRandom(data.timeA, clusters.nTracks);
  clusters.timeA = data.timeA.data();
  fillRandom(data.padA, clusters.nTracks);
  clusters.padA = data.padA.data();
  fillRandom(data.nTrackClusters, clusters.nTracks);
  clusters.nTrackClusters = data.nTrackClusters.data();

  clusters.nUnattachedClusters = rand() % 32;
  fillRandom(data.qTotU, clusters.nUnattachedClusters);
  clusters.qTotU = data.qTotU.data();
  fillRandom(data.qMaxU, clusters.nUnattachedClusters);
  clusters.qMaxU = data.qMaxU.data();
  fillRandom(data.flagsU, clusters.nUnattachedClusters);
  clusters.flagsU = data.flagsU.data();
  fillRandom(data.padDiffU, clusters.nUnattachedClusters);
  clusters.padDiffU = data.padDiffU.data();
  fillRandom(data.timeDiffU, clusters.nUnattachedClusters);
  clusters.timeDiffU = data.timeDiffU.data();
  fillRandom(data.sigmaPadU, clusters.nUnattachedClusters);
  clusters.sigmaPadU = data.sigmaPadU.data();
  fillRandom(data.sigmaTimeU, clusters.nUnattachedClusters);
  clusters.sigmaTimeU = data.sigmaTimeU.data();

  clusters.nSliceRows = rand() % 32;
  fillRandom(data.nSliceRowClusters, clusters.nSliceRows);
  clusters.nSliceRowClusters = data.nSliceRowClusters.data();
}

BOOST_AUTO_TEST_CASE(test_tpc_compressedclusters)
{
  CompressedClustersROOT clusters;
  ClustersData data;
  fillClusters(clusters, data);
  std::vector<char> buffer;
  CompressedClustersHelpers::flattenTo(buffer, clusters);
  CompressedClustersROOT restored;
  CompressedClustersCounters& x = restored;
  x = static_cast<CompressedClustersCounters&>(clusters);
  BOOST_REQUIRE(restored.qTotA == nullptr);
  CompressedClustersHelpers::restoreFrom(buffer, restored);

  // check one entry from each category
  BOOST_CHECK(restored.nAttachedClusters == data.qMaxA.size());
  BOOST_CHECK(memcmp(restored.qMaxA, data.qMaxA.data(), restored.nAttachedClusters * sizeof(decltype(data.qMaxA)::value_type)) == 0);
  BOOST_CHECK(restored.nTracks == data.nTrackClusters.size());
  BOOST_CHECK(memcmp(restored.nTrackClusters, data.nTrackClusters.data(), restored.nTracks * sizeof(decltype(data.nTrackClusters)::value_type)) == 0);
  BOOST_CHECK(restored.nUnattachedClusters == data.qMaxU.size());
  BOOST_CHECK(memcmp(restored.qMaxU, data.qMaxU.data(), restored.nUnattachedClusters * sizeof(decltype(data.qMaxU)::value_type)) == 0);
  BOOST_CHECK(restored.nSliceRows == data.nSliceRowClusters.size());
  BOOST_CHECK(memcmp(restored.nSliceRowClusters, data.nSliceRowClusters.data(), restored.nSliceRows * sizeof(decltype(data.nSliceRowClusters)::value_type)) == 0);
}

BOOST_AUTO_TEST_CASE(test_tpc_compressedclusters_root_streaming)
{
  CompressedClustersROOT clusters;
  ClustersData data;
  fillClusters(clusters, data);

  std::string fileName = gSystem->TempDirectory();
  fileName += "/testCompressedClusters.root";

  { // scope for the creation of the test file
    std::unique_ptr<TFile> testFile(TFile::Open(fileName.c_str(), "RECREATE"));
    std::unique_ptr<TTree> testTree = std::make_unique<TTree>("testtree", "testtree");

    auto* branch = testTree->Branch("compclusters", &clusters);
    testTree->Fill();
    testTree->Write();
    testTree->SetDirectory(nullptr);
    testFile->Close();
  }

  { // scope for read back of the streamed object
    std::unique_ptr<TFile> file(TFile::Open(fileName.c_str()));
    BOOST_REQUIRE(file != nullptr);
    TTree* tree = reinterpret_cast<TTree*>(file->GetObjectChecked("testtree", "TTree"));
    BOOST_REQUIRE(tree != nullptr);
    TBranch* branch = tree->GetBranch("compclusters");
    BOOST_REQUIRE(branch != nullptr);
    CompressedClustersROOT* readback = nullptr;
    branch->SetAddress(&readback);
    branch->GetEntry(0);
    BOOST_REQUIRE(readback != nullptr);

    CompressedClustersROOT& restored = *readback;
    // check one entry from each category
    BOOST_CHECK(restored.nAttachedClusters == data.qMaxA.size());
    BOOST_CHECK(memcmp(restored.qMaxA, data.qMaxA.data(), restored.nAttachedClusters * sizeof(decltype(data.qMaxA)::value_type)) == 0);
    BOOST_CHECK(restored.nTracks == data.nTrackClusters.size());
    BOOST_CHECK(memcmp(restored.nTrackClusters, data.nTrackClusters.data(), restored.nTracks * sizeof(decltype(data.nTrackClusters)::value_type)) == 0);
    BOOST_CHECK(restored.nUnattachedClusters == data.qMaxU.size());
    BOOST_CHECK(memcmp(restored.qMaxU, data.qMaxU.data(), restored.nUnattachedClusters * sizeof(decltype(data.qMaxU)::value_type)) == 0);
    BOOST_CHECK(restored.nSliceRows == data.nSliceRowClusters.size());
    BOOST_CHECK(memcmp(restored.nSliceRowClusters, data.nSliceRowClusters.data(), restored.nSliceRows * sizeof(decltype(data.nSliceRowClusters)::value_type)) == 0);
  }
}

} // namespace o2::tpc
