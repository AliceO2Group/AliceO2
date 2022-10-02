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

/// @file ClusterNativeHelper.cxx
/// @brief Helper class to read the binary format of TPC ClusterNative
/// @since 2019-01-23
/// @author Matthias Richter

#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "Algorithm/Parser.h"
#include <TBranch.h>
#include <fairlogger/Logger.h>
#include <iostream>

using namespace o2::tpc;
using namespace o2::tpc::constants;

void ClusterNativeHelper::convert(const char* fromFile, const char* toFile, const char* toTreeName)
{
  Reader reader;
  TreeWriter writer;
  reader.init(fromFile);
  writer.init(toFile, toTreeName);
  size_t nEntries = reader.getTreeSize();
  ClusterNativeAccess clusterIndex;
  std::unique_ptr<ClusterNative[]> clusterBuffer;
  ConstMCLabelContainerViewWithBuffer mcBuffer;

  int result = 0;
  int nClusters = 0;
  for (size_t entry = 0; entry < nEntries; ++entry) {
    LOG(info) << "converting entry " << entry;
    reader.read(entry);
    result = reader.fillIndex(clusterIndex, clusterBuffer, mcBuffer);
    if (result >= 0) {
      LOG(info) << "added " << result << " clusters to index";
    } else {
      LOG(error) << "filling of clusters index failed with " << result;
    }
    result = writer.fillFrom(clusterIndex);
    if (result >= 0) {
      LOG(info) << "wrote " << result << " clusters to tree";
      nClusters += result;
    } else {
      LOG(error) << "filling of tree failed with " << result;
    }
  }
  LOG(info) << "... done, converted " << nClusters << " clusters";
  writer.close();
}

std::unique_ptr<ClusterNativeAccess> ClusterNativeHelper::createClusterNativeIndex(
  std::unique_ptr<ClusterNative[]>& buffer, std::vector<ClusterNativeContainer>& clusters,
  MCLabelContainer* bufferMC, std::vector<MCLabelContainer>* mcTruth)
{
  std::unique_ptr<ClusterNativeAccess> retVal(new ClusterNativeAccess);
  memset(retVal.get(), 0, sizeof(*retVal));
  for (int i = 0; i < clusters.size(); i++) {
    if (retVal->nClusters[clusters[i].sector][clusters[i].globalPadRow]) {
      LOG(error) << "Received two containers for the same sector / row";
      return std::unique_ptr<ClusterNativeAccess>();
    }
    retVal->nClusters[clusters[i].sector][clusters[i].globalPadRow] = clusters[i].clusters.size();
    retVal->nClustersTotal += clusters[i].clusters.size();
  }
  buffer.reset(new ClusterNative[retVal->nClustersTotal]);
  if (bufferMC) {
    bufferMC->clear();
  }
  retVal->clustersLinear = buffer.get();
  retVal->setOffsetPtrs();
  for (int i = 0; i < clusters.size(); i++) {
    memcpy(&buffer[retVal->clusterOffset[clusters[i].sector][clusters[i].globalPadRow]], clusters[i].clusters.data(), sizeof(*retVal->clustersLinear) * clusters[i].clusters.size());
    if (mcTruth) {
      for (unsigned int j = 0; j < clusters[i].clusters.size(); j++) {
        for (auto const& label : (*mcTruth)[i].getLabels(j)) {
          bufferMC->addElement(retVal->clusterOffset[clusters[i].sector][clusters[i].globalPadRow] + j, label);
        }
      }
    }
  }
  return (std::move(retVal));
}

ClusterNativeHelper::Reader::~Reader()
{
  clear();
}

ClusterNativeHelper::Reader::Reader()
{
  memset(&mSectorRawSize, 0, sizeof(mSectorRawSize));
  memset(&mSectorRaw, 0, sizeof(mSectorRaw));
}

void ClusterNativeHelper::Reader::init(const char* filename, const char* treename)
{
  if (treename != nullptr && treename[0] != 0) {
    mTreeName = treename;
  }
  mFile.reset(TFile::Open(filename));
  if (!mFile) {
    return;
  }
  mTree = reinterpret_cast<TTree*>(mFile->GetObjectUnchecked(mTreeName.c_str()));
  if (!mTree) {
    LOG(error) << "can not find tree " << mTreeName << " in file " << filename;
    return;
  }

  const bool singleBranch = mTree->GetBranch(mDataBranchName.data());

  size_t nofDataBranches = 0;
  size_t nofMCBranches = 0;
  for (size_t sector = 0; sector < NSectors; ++sector) {
    auto branchname = singleBranch ? mDataBranchName : mDataBranchName + "_" + std::to_string(sector);
    TBranch* branch = mTree->GetBranch(branchname.c_str());
    if (branch) {
      TBranch* sizebranch = mTree->GetBranch((branchname + "Size").c_str());
      if (sizebranch) {
        branch->SetAddress(&mSectorRaw[sector]);
        sizebranch->SetAddress(&mSectorRawSize[sector]);
        ++nofDataBranches;
      } else {
        LOG(error) << "can not find corresponding 'Size' branch for data branch " << branchname << ", skipping it";
      }
    }
    branchname = singleBranch ? mMCBranchName : mMCBranchName + "_" + std::to_string(sector);
    branch = mTree->GetBranch(branchname.c_str());
    if (branch) {
      branch->SetAddress(&mSectorMCPtr[sector]);
      ++nofMCBranches;
    }

    if (singleBranch) {
      break;
    }
  }
  LOG(info) << "reading " << nofDataBranches << " data branch(es) and " << nofMCBranches << " mc branch(es)";
}

void ClusterNativeHelper::Reader::read(size_t entry)
{
  if (entry >= getTreeSize()) {
    return;
  }
  clear();
  mTree->GetEntry(entry);
}

void ClusterNativeHelper::Reader::clear()
{
  memset(&mSectorRawSize, 0, sizeof(mSectorRawSize));
  for (auto data : mSectorRaw) {
    if (data) {
      delete data;
    }
  }
  memset(&mSectorRaw, 0, sizeof(mSectorRaw));
}

int ClusterNativeHelper::Reader::fillIndex(ClusterNativeAccess& clusterIndex, std::unique_ptr<ClusterNative[]>& clusterBuffer,
                                           ConstMCLabelContainerViewWithBuffer& mcBuffer)
{
  std::vector<gsl::span<const char>> clustersTPC;
  std::vector<ConstMCLabelContainer> constMCLabelContainers;
  std::vector<ConstMCLabelContainerView> constMCLabelContainerViews;

  for (size_t index = 0; index < mSectorRaw.size(); ++index) {
    if (mSectorRaw[index]) {
      if (mSectorRaw[index]->size() != mSectorRawSize[index]) {
        LOG(error) << "inconsistent raw size for sector " << index << ": " << mSectorRaw[index]->size() << " v.s. " << mSectorRawSize[index];
        mSectorRaw[index]->clear();
      } else {
        clustersTPC.emplace_back(mSectorRaw[index]->data(), mSectorRawSize[index]);
      }
    }
    if (mSectorMCPtr[index]) {
      auto& view = constMCLabelContainers.emplace_back();
      mSectorMCPtr[index]->copyandflatten(view);
      constMCLabelContainerViews.emplace_back(view);
    }
  }

  int result = fillIndex(clusterIndex, clusterBuffer, mcBuffer, clustersTPC, constMCLabelContainerViews);
  return result;
}

int ClusterNativeHelper::Reader::parseSector(const char* buffer, size_t size, gsl::span<ConstMCLabelContainerView const> const& mcinput, ClusterNativeAccess& clusterIndex,
                                             const ConstMCLabelContainerView* (&clustersMCTruth)[MAXSECTOR])
{
  if (!buffer || size < sizeof(ClusterCountIndex)) {
    return 0;
  }

  auto mcIterator = mcinput.begin();
  ClusterCountIndex const& counts = *reinterpret_cast<const ClusterCountIndex*>(buffer);
  ClusterNative const* clusters = reinterpret_cast<ClusterNative const*>(buffer + sizeof(ClusterCountIndex));
  size_t numberOfClusters = 0;
  for (int i = 0; i < MAXSECTOR; i++) {
    int nSectorClusters = 0;
    for (int j = 0; j < MAXGLOBALPADROW; j++) {
      if (counts.nClusters[i][j] == 0) {
        continue;
      }
      nSectorClusters += counts.nClusters[i][j];
      if ((numberOfClusters + counts.nClusters[i][j]) * sizeof(ClusterNative) + sizeof(ClusterCountIndex) > size) {
        throw std::runtime_error("inconsistent buffer size");
      }
      clusterIndex.clusters[i][j] = clusters + numberOfClusters;
      clusterIndex.nClusters[i][j] = counts.nClusters[i][j];
      numberOfClusters += counts.nClusters[i][j];
    }
    if (nSectorClusters > 0) {
      if (mcIterator != mcinput.end()) {
        clustersMCTruth[i] = &(*mcIterator);
        ++mcIterator;
        if (mcIterator != mcinput.end()) {
          throw std::runtime_error("can only have one MCLabel block per sector");
        }
      }
    }
  }

  return numberOfClusters;
}

ClusterNativeHelper::TreeWriter::~TreeWriter()
{
  close();
}

void ClusterNativeHelper::TreeWriter::init(const char* filename, const char* treename)
{
  mFile.reset(TFile::Open(filename, "RECREATE"));
  if (!mFile) {
    return;
  }
  mTree = std::make_unique<TTree>(treename, treename);
  if (!mTree) {
    return;
  }

  mTree->Branch("event", &mEvent, "Event/I");
  mTree->Branch("NativeClusters", "std::vector<o2::tpc::ClusterNativeHelper::TreeWriter::BranchData>", &mStore);
  mEvent = 0;
  mStoreClusters.clear();
}

int ClusterNativeHelper::TreeWriter::fillFrom(ClusterNativeAccess const& clusterIndex)
{
  if (!mTree) {
    return -1;
  }
  int result = 0;
  for (size_t sector = 0; sector < MAXSECTOR; ++sector) {
    for (size_t padrow = 0; padrow < MAXGLOBALPADROW; ++padrow) {
      int locres = fillFrom(sector, padrow, clusterIndex.clusters[sector][padrow], clusterIndex.nClusters[sector][padrow]);
      if (result >= 0 && locres >= 0) {
        result += locres;
      } else if (result >= 0) {
        result = locres;
      }
    }
  }
  ++mEvent;
  return result;
}

int ClusterNativeHelper::TreeWriter::fillFrom(int sector, int padrow, ClusterNative const* clusters, size_t nClusters, MCLabelContainer*)
{
  if (!mTree) {
    return -1;
  }
  mStoreClusters.resize(nClusters);
  if (clusters != nullptr && nClusters > 0) {
    std::fill(mStoreClusters.begin(), mStoreClusters.end(), BranchData{sector, padrow});
    std::copy(clusters, clusters + nClusters, mStoreClusters.begin());
    mTree->Fill();
  }
  return nClusters;
}

void ClusterNativeHelper::TreeWriter::close()
{
  if (!mFile) {
    return;
  }
  mFile->Write();
  mFile->Close();
  mTree.release();
  mFile.reset();
}
