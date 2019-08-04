// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <FairLogger.h>
#include <iostream>

using namespace o2::tpc;

void ClusterNativeHelper::convert(const char* fromFile, const char* toFile, const char* toTreeName)
{
  Reader reader;
  TreeWriter writer;
  reader.init(fromFile);
  writer.init(toFile, toTreeName);
  size_t nEntries = reader.getTreeSize();
  ClusterNativeAccess clusterIndex;
  std::unique_ptr<ClusterNative[]> clusterBuffer;
  MCLabelContainer mcBuffer;

  int result = 0;
  int nClusters = 0;
  for (size_t entry = 0; entry < nEntries; ++entry) {
    LOG(INFO) << "converting entry " << entry;
    reader.read(entry);
    result = reader.fillIndex(clusterIndex, clusterBuffer, mcBuffer);
    if (result >= 0) {
      LOG(INFO) << "added " << result << " clusters to index";
    } else {
      LOG(ERROR) << "filling of clusters index failed with " << result;
    }
    result = writer.fillFrom(clusterIndex);
    if (result >= 0) {
      LOG(INFO) << "wrote " << result << " clusters to tree";
      nClusters += result;
    } else {
      LOG(ERROR) << "filling of tree failed with " << result;
    }
  }
  LOG(INFO) << "... done, converted " << nClusters << " clusters";
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
      LOG(ERROR) << "Received two containers for the same sector / row";
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
    LOG(ERROR) << "can not find tree " << mTreeName << " in file " << filename;
    return;
  }
  size_t nofDataBranches = 0;
  size_t nofMCBranches = 0;
  for (size_t sector = 0; sector < NSectors; ++sector) {
    auto branchname = mDataBranchName + "_" + std::to_string(sector);
    TBranch* branch = mTree->GetBranch(branchname.c_str());
    if (branch) {
      TBranch* sizebranch = mTree->GetBranch((branchname + "Size").c_str());
      if (sizebranch) {
        branch->SetAddress(&mSectorRaw[sector]);
        sizebranch->SetAddress(&mSectorRawSize[sector]);
        ++nofDataBranches;
      } else {
        LOG(ERROR) << "can not find corresponding 'Size' branch for data branch " << branchname << ", skipping it";
      }
    }
    branchname = mMCBranchName + "_" + std::to_string(sector);
    branch = mTree->GetBranch(branchname.c_str());
    if (branch) {
      mSectorMCPtr[sector] = &mSectorMC[sector];
      branch->SetAddress(&mSectorMCPtr[sector]);
      ++nofMCBranches;
    }
  }
  LOG(INFO) << "reading " << nofDataBranches << " data branch(es) and " << nofMCBranches << " mc branch(es)";
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
                                           MCLabelContainer& mcBuffer)
{
  for (size_t index = 0; index < mSectorRaw.size(); ++index) {
    if (mSectorRaw[index] && mSectorRaw[index]->size() != mSectorRawSize[index]) {
      LOG(ERROR) << "inconsistent raw size for sector " << index << ": " << mSectorRaw[index]->size() << " v.s. " << mSectorRawSize[index];
      mSectorRaw[index]->clear();
    }
  }
  int result = fillIndex(clusterIndex, clusterBuffer, mcBuffer, mSectorRaw, mSectorMC, [](auto&) { return true; });
  return result;
}

int ClusterNativeHelper::Reader::parseSector(const char* buffer, size_t size, std::vector<MCLabelContainer>& mcinput, ClusterNativeAccess& clusterIndex,
                                             const MCLabelContainer* (&clustersMCTruth)[Constants::MAXSECTOR][Constants::MAXGLOBALPADROW])
{
  if (!buffer || size == 0) {
    return 0;
  }

  auto mcIterator = mcinput.begin();
  using ClusterGroupParser = o2::algorithm::ForwardParser<o2::tpc::ClusterGroupHeader>;
  ClusterGroupParser parser;
  size_t numberOfClusters = 0;
  parser.parse(
    buffer, size,
    [](const typename ClusterGroupParser::HeaderType& h) {
      // check the header, but in this case there is no validity check
      return true;
    },
    [](const typename ClusterGroupParser::HeaderType& h) {
      // get the size of the frame including payload
      // and header and trailer size, e.g. payload size
      // from a header member
      return h.nClusters * sizeof(ClusterNative) + ClusterGroupParser::totalOffset;
    },
    [&](typename ClusterGroupParser::FrameInfo& frame) {
      int sector = frame.header->sector;
      int padrow = frame.header->globalPadRow;
      int nClusters = frame.header->nClusters;
      clusterIndex.clusters[sector][padrow] = reinterpret_cast<const ClusterNative*>(frame.payload);
      clusterIndex.nClusters[sector][padrow] = nClusters;
      numberOfClusters += nClusters;
      if (mcIterator != mcinput.end()) {
        clustersMCTruth[sector][padrow] = &(*mcIterator);
        ++mcIterator;
      }

      return true;
    });
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
  for (size_t sector = 0; sector < Constants::MAXSECTOR; ++sector) {
    for (size_t padrow = 0; padrow < Constants::MAXGLOBALPADROW; ++padrow) {
      int locres = fillFrom(sector, padrow, clusterIndex.clusters[sector][padrow], clusterIndex.nClusters[sector][padrow]);
      if (result >= 0 && locres >= 0) {
        result += locres;
      } else if (result >= 0) {
        result = locres;
      }
    }
  }
  return result;
}

int ClusterNativeHelper::TreeWriter::fillFrom(int sector, int padrow, ClusterNative const* clusters, size_t nClusters, MCLabelContainer*)
{
  if (!mTree) {
    return -1;
  }
  mStoreClusters.resize(nClusters, BranchData{sector, padrow});
  if (clusters != nullptr && nClusters > 0) {
    std::copy(clusters, clusters + nClusters, mStoreClusters.begin());
  }
  mTree->Fill();
  ++mEvent;
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
