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

/// \file TopologyDictionary.cxx

#include "ITS3Reconstruction/BuildTopologyDictionary.h"
#include "ITSMFTReconstruction/LookUp.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITSMFTBase/SegmentationAlpide.h"

namespace o2::its3
{

void BuildTopologyDictionary::accountTopology(const itsmft::ClusterTopology& cluster, float dX, float dZ)
{
  mTotClusters++;
  bool useDf = dX < IgnoreVal / 2; // we may need to account the frequency but to not update the centroid
  // std::pair<unordered_map<unsigned long, TopoStat>::iterator,bool> ret;
  //       auto ret = mTopologyMap.insert(std::make_pair(cluster.getHash(), std::make_pair(cluster, 1)));

  auto& topoStat = mTopologyMap[cluster.getHash()];
  topoStat.countsTotal++;
  if (topoStat.countsTotal == 1) { // a new topology is inserted
    topoStat.topology = cluster;
    //___________________DEFINING_TOPOLOGY_CHARACTERISTICS__________________
    itsmft::TopologyInfo topInf;
    topInf.mPattern.setPattern(cluster.getPattern().data());
    int& rs = topInf.mSizeX = cluster.getRowSpan();
    int& cs = topInf.mSizeZ = cluster.getColumnSpan();
    //__________________COG_Determination_____________
    topInf.mNpixels = cluster.getClusterPattern().getCOG(topInf.mCOGx, topInf.mCOGz);
    if (useDf) {
      topInf.mXmean = dX;
      topInf.mZmean = dZ;
      topoStat.countsWithBias = 1;
    } else { // assign expected sigmas from the pixel X, Z sizes
      topInf.mXsigma2 = 1 / 12. / std::min(10, topInf.mSizeX);
      topInf.mZsigma2 = 1 / 12. / std::min(10, topInf.mSizeZ);
    }
    mMapInfo.insert(std::make_pair(cluster.getHash(), topInf));
  } else {
    if (useDf) {
      auto num = topoStat.countsWithBias++;
      auto ind = mMapInfo.find(cluster.getHash());
      float tmpxMean = ind->second.mXmean;
      float newxMean = ind->second.mXmean = ((tmpxMean)*num + dX) / (num + 1);
      float tmpxSigma2 = ind->second.mXsigma2;
      ind->second.mXsigma2 = (num * tmpxSigma2 + (dX - tmpxMean) * (dX - newxMean)) / (num + 1); // online variance algorithm
      float tmpzMean = ind->second.mZmean;
      float newzMean = ind->second.mZmean = ((tmpzMean)*num + dZ) / (num + 1);
      float tmpzSigma2 = ind->second.mZsigma2;
      ind->second.mZsigma2 = (num * tmpzSigma2 + (dZ - tmpzMean) * (dZ - newzMean)) / (num + 1); // online variance algorithm
    }
  }
}

void BuildTopologyDictionary::groupRareTopologies()
{
  std::cout << "Dictionary finalisation" << std::endl;
  std::cout << "Number of clusters: " << mTotClusters << std::endl;

  double totFreq = 0.;
  for (int j = 0; j < mNCommonTopologies; j++) {
    itsmft::GroupStruct gr;
    gr.mHash = mTopologyFrequency[j].second;
    gr.mFrequency = ((double)(mTopologyFrequency[j].first)) / mTotClusters;
    totFreq += gr.mFrequency;
    // rough estimation for the error considering a8 uniform distribution
    gr.mErrX = std::sqrt(mMapInfo.find(gr.mHash)->second.mXsigma2);
    gr.mErrZ = std::sqrt(mMapInfo.find(gr.mHash)->second.mZsigma2);
    gr.mErr2X = mMapInfo.find(gr.mHash)->second.mXsigma2;
    gr.mErr2Z = mMapInfo.find(gr.mHash)->second.mZsigma2;
    gr.mXCOG = -1 * mMapInfo.find(gr.mHash)->second.mCOGx;
    gr.mZCOG = mMapInfo.find(gr.mHash)->second.mCOGz;
    gr.mNpixels = mMapInfo.find(gr.mHash)->second.mNpixels;
    gr.mPattern = mMapInfo.find(gr.mHash)->second.mPattern;
    gr.mIsGroup = false;
    mDictionary.mVectorOfIDs.push_back(gr);
    if (j == int(itsmft::CompCluster::InvalidPatternID - 1)) {
      LOGP(warning, "Limiting N unique topologies to {}, threshold freq. to {}, cumulative freq. to {} to be below InvalidPatternID", j, gr.mFrequency, totFreq);
      mNCommonTopologies = j;
      mFrequencyThreshold = gr.mFrequency;
      break;
    }
  }
  // groupRareTopologies based on binning over number of rows and columns (TopologyDictionary::NumberOfRowClasses *
  // NumberOfColClasses)

  std::unordered_map<int, std::pair<itsmft::GroupStruct, unsigned long>> tmp_GroupMap; //<group ID, <Group struct, counts>>

  int grNum = 0;
  int rowBinEdge = 0;
  int colBinEdge = 0;
  for (int iRowClass = 0; iRowClass < itsmft::TopologyDictionary::MaxNumberOfRowClasses; iRowClass++) {
    for (int iColClass = 0; iColClass < itsmft::TopologyDictionary::MaxNumberOfColClasses; iColClass++) {
      rowBinEdge = (iRowClass + 1) * itsmft::TopologyDictionary::RowClassSpan;
      colBinEdge = (iColClass + 1) * itsmft::TopologyDictionary::ColClassSpan;
      grNum = itsmft::LookUp::groupFinder(rowBinEdge, colBinEdge);
      // Create a structure for a group of rare topologies
      itsmft::GroupStruct gr;
      gr.mHash = (((unsigned long)(grNum)) << 32) & 0xffffffff00000000;
      gr.mErrX = itsmft::TopologyDictionary::RowClassSpan / std::sqrt(12 * std::min(10, rowBinEdge));
      gr.mErrZ = itsmft::TopologyDictionary::ColClassSpan / std::sqrt(12 * std::min(10, colBinEdge));
      gr.mErr2X = gr.mErrX * gr.mErrX;
      gr.mErr2Z = gr.mErrZ * gr.mErrZ;
      gr.mXCOG = 0;
      gr.mZCOG = 0;
      gr.mNpixels = rowBinEdge * colBinEdge;
      gr.mIsGroup = true;
      gr.mFrequency = 0.;
      /// A dummy pattern with all fired pixels in the bounding box is assigned to groups of rare topologies.
      unsigned char dummyPattern[itsmft::ClusterPattern::kExtendedPatternBytes] = {0};
      dummyPattern[0] = (unsigned char)rowBinEdge;
      dummyPattern[1] = (unsigned char)colBinEdge;
      int nBits = rowBinEdge * colBinEdge;
      int nBytes = nBits / 8;
      for (int iB = 2; iB < nBytes + 2; iB++) {
        dummyPattern[iB] = (unsigned char)255;
      }
      int residualBits = nBits % 8;
      if (residualBits) {
        unsigned char tempChar = 0;
        while (residualBits > 0) {
          residualBits--;
          tempChar |= 1 << (7 - residualBits);
        }
        dummyPattern[nBytes + 2] = tempChar;
      }
      gr.mPattern.setPattern(dummyPattern);
      // Filling the map for groups
      tmp_GroupMap[grNum] = std::make_pair(gr, 0);
    }
  }
  int rs;
  int cs;
  int index;

  // Updating the counts for the groups of rare topologies
  for (unsigned int j = (unsigned int)mNCommonTopologies; j < mTopologyFrequency.size(); j++) {
    unsigned long hash1 = mTopologyFrequency[j].second;
    rs = mTopologyMap.find(hash1)->second.topology.getRowSpan();
    cs = mTopologyMap.find(hash1)->second.topology.getColumnSpan();
    index = itsmft::LookUp::groupFinder(rs, cs);
    tmp_GroupMap[index].second += mTopologyFrequency[j].first;
  }

  for (auto&& p : tmp_GroupMap) {
    itsmft::GroupStruct& group = p.second.first;
    group.mFrequency = ((double)p.second.second) / mTotClusters;
    mDictionary.mVectorOfIDs.push_back(group);
  }

  // Sorting the dictionary preserving all unique topologies
  std::sort(mDictionary.mVectorOfIDs.begin(), mDictionary.mVectorOfIDs.end(), [](const itsmft::GroupStruct& a, const itsmft::GroupStruct& b) {
    return (!a.mIsGroup) && b.mIsGroup ? true : (a.mIsGroup && (!b.mIsGroup) ? false : (a.mFrequency > b.mFrequency));
  });
  if (mDictionary.mVectorOfIDs.size() >= itsmft::CompCluster::InvalidPatternID - 1) {
    LOGP(warning, "Max allowed {} patterns is reached, stopping", itsmft::CompCluster::InvalidPatternID - 1);
    mDictionary.mVectorOfIDs.resize(itsmft::CompCluster::InvalidPatternID - 1);
  }
  // Sorting the dictionary to final form
  std::sort(mDictionary.mVectorOfIDs.begin(), mDictionary.mVectorOfIDs.end(), [](const itsmft::GroupStruct& a, const itsmft::GroupStruct& b) { return a.mFrequency > b.mFrequency; });
  // Creating the map for common topologies
  for (int iKey = 0; iKey < mDictionary.getSize(); iKey++) {
    itsmft::GroupStruct& gr = mDictionary.mVectorOfIDs[iKey];
    if (!gr.mIsGroup) {
      mDictionary.mCommonMap.insert(std::make_pair(gr.mHash, iKey));
      if (gr.mPattern.getUsedBytes() == 1) {
        mDictionary.mSmallTopologiesLUT[(gr.mPattern.getColumnSpan() - 1) * 255 + (int)gr.mPattern.getByte(2)] = iKey;
      }
    } else {
      mDictionary.mGroupMap.insert(std::make_pair((int)(gr.mHash >> 32) & 0x00000000ffffffff, iKey));
    }
  }
  std::cout << "Dictionay finalised" << std::endl;
  std::cout << "Number of keys: " << mDictionary.getSize() << std::endl;
  std::cout << "Number of common topologies: " << mDictionary.mCommonMap.size() << std::endl;
  std::cout << "Number of groups of rare topologies: " << mDictionary.mGroupMap.size() << std::endl;
}

} // namespace o2::its3
