// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file BuildTopologyDictionary.cxx
/// \brief Implementation of the BuildTopologyDictionary class.
///
/// \author Luca Barioglio, University and INFN of Torino

#include "ITSMFTReconstruction/BuildTopologyDictionary.h"
#include "ITSMFTReconstruction/LookUp.h"
#include <cmath>
#include <TFile.h>

ClassImp(o2::itsmft::BuildTopologyDictionary);

namespace o2
{
namespace itsmft
{
constexpr float BuildTopologyDictionary::IgnoreVal;

BuildTopologyDictionary::BuildTopologyDictionary() : mTotClusters{0} {}

void BuildTopologyDictionary::accountTopology(const ClusterTopology& cluster, float dX, float dZ)
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
    TopologyInfo topInf;
    topInf.mPattern.setPattern(cluster.getPattern().data());
    int& rs = topInf.mSizeX = cluster.getRowSpan();
    int& cs = topInf.mSizeZ = cluster.getColumnSpan();
    //__________________COG_Determination_____________
    topInf.mNpixels = cluster.getClusterPattern().getCOG(topInf.mCOGx, topInf.mCOGz);
    if (useDf) {
      topInf.mXmean = dX;
      topInf.mZmean = dZ;
      topoStat.countsWithBias = 1;
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

void BuildTopologyDictionary::setThreshold(double thr)
{
  mTopologyFrequency.clear();
  for (auto&& p : mTopologyMap) { // p is pair<ulong,TopoStat>
    mTopologyFrequency.emplace_back(std::make_pair(p.second.countsTotal, p.first));
  }
  std::sort(mTopologyFrequency.begin(), mTopologyFrequency.end(),
            [](const std::pair<unsigned long, unsigned long>& couple1,
               const std::pair<unsigned long, unsigned long>& couple2) { return (couple1.first > couple2.first); });
  mNCommonTopologies = 0;
  mDictionary.mCommonMap.clear();
  mDictionary.mGroupMap.clear();
  mFrequencyThreshold = thr;
  for (auto& q : mTopologyFrequency) {
    if (((double)q.first) / mTotClusters > thr) {
      mNCommonTopologies++;
    } else {
      break;
    }
  }
}

void BuildTopologyDictionary::setNCommon(unsigned int nCommon)
{
  mTopologyFrequency.clear();
  for (auto&& p : mTopologyMap) { // p os pair<ulong,TopoStat>
    mTopologyFrequency.emplace_back(std::make_pair(p.second.countsTotal, p.first));
  }
  std::sort(mTopologyFrequency.begin(), mTopologyFrequency.end(),
            [](const std::pair<unsigned long, unsigned long>& couple1,
               const std::pair<unsigned long, unsigned long>& couple2) { return (couple1.first > couple2.first); });
  mNCommonTopologies = nCommon;
  mDictionary.mCommonMap.clear();
  mDictionary.mGroupMap.clear();
  mFrequencyThreshold = ((double)mTopologyFrequency[mNCommonTopologies - 1].first) / mTotClusters;
}

void BuildTopologyDictionary::setThresholdCumulative(double cumulative)
{
  mTopologyFrequency.clear();
  if (cumulative <= 0. || cumulative >= 1.) {
    cumulative = 0.99;
  }
  double totFreq = 0.;
  for (auto&& p : mTopologyMap) { // p os pair<ulong,TopoStat>
    mTopologyFrequency.emplace_back(std::make_pair(p.second.countsTotal, p.first));
  }
  std::sort(mTopologyFrequency.begin(), mTopologyFrequency.end(),
            [](const std::pair<unsigned long, unsigned long>& couple1,
               const std::pair<unsigned long, unsigned long>& couple2) { return (couple1.first > couple2.first); });
  mNCommonTopologies = 0;
  mDictionary.mCommonMap.clear();
  mDictionary.mGroupMap.clear();
  for (auto& q : mTopologyFrequency) {
    totFreq += ((double)(q.first)) / mTotClusters;
    if (totFreq < cumulative) {
      mNCommonTopologies++;
    } else {
      break;
    }
  }
  mFrequencyThreshold = ((double)(mTopologyFrequency[--mNCommonTopologies].first)) / mTotClusters;
  while (std::fabs(((double)mTopologyFrequency[mNCommonTopologies].first) / mTotClusters - mFrequencyThreshold) < 1.e-15) {
    mNCommonTopologies--;
  }
  mFrequencyThreshold = ((double)mTopologyFrequency[mNCommonTopologies++].first) / mTotClusters;
}

void BuildTopologyDictionary::groupRareTopologies()
{
  std::cout << "Dictionary finalisation" << std::endl;
  std::cout << "Number of clusters: " << mTotClusters << std::endl;

  double totFreq = 0.;
  for (int j = 0; j < mNCommonTopologies; j++) {
    GroupStruct gr;
    gr.mHash = mTopologyFrequency[j].second;
    gr.mFrequency = ((double)(mTopologyFrequency[j].first)) / mTotClusters;
    // rough estimation for the error considering a8 uniform distribution
    gr.mErrX = std::sqrt(mMapInfo.find(gr.mHash)->second.mXsigma2);
    gr.mErrZ = std::sqrt(mMapInfo.find(gr.mHash)->second.mZsigma2);
    gr.mErr2X = gr.mErrX * gr.mErrX;
    gr.mErr2Z = gr.mErrZ * gr.mErrZ;
    gr.mXCOG = -1 * mMapInfo.find(gr.mHash)->second.mCOGx * o2::itsmft::SegmentationAlpide::PitchRow;
    gr.mZCOG = mMapInfo.find(gr.mHash)->second.mCOGz * o2::itsmft::SegmentationAlpide::PitchCol;
    gr.mNpixels = mMapInfo.find(gr.mHash)->second.mNpixels;
    gr.mPattern = mMapInfo.find(gr.mHash)->second.mPattern;
    gr.mIsGroup = false;
    mDictionary.mVectorOfIDs.push_back(gr);
  }
  // groupRareTopologies based on binning over number of rows and columns (TopologyDictionary::NumberOfRowClasses *
  // NumberOfColClasses)

  std::unordered_map<int, std::pair<GroupStruct, unsigned long>> tmp_GroupMap; //<group ID, <Group struct, counts>>

  int grNum = 0;
  int rowBinEdge = 0;
  int colBinEdge = 0;
  for (int iRowClass = 0; iRowClass < TopologyDictionary::MaxNumberOfRowClasses; iRowClass++) {
    for (int iColClass = 0; iColClass < TopologyDictionary::MaxNumberOfColClasses; iColClass++) {
      rowBinEdge = (iRowClass + 1) * TopologyDictionary::RowClassSpan;
      colBinEdge = (iColClass + 1) * TopologyDictionary::ColClassSpan;
      grNum = LookUp::groupFinder(rowBinEdge, colBinEdge);
      // Create a structure for a group of rare topologies
      GroupStruct gr;
      gr.mHash = (((unsigned long)(grNum)) << 32) & 0xffffffff00000000;
      gr.mErrX = (rowBinEdge)*o2::itsmft::SegmentationAlpide::PitchRow / std::sqrt(12);
      gr.mErrZ = (colBinEdge)*o2::itsmft::SegmentationAlpide::PitchCol / std::sqrt(12);
      gr.mErr2X = gr.mErrX * gr.mErrX;
      gr.mErr2Z = gr.mErrZ * gr.mErrZ;
      gr.mXCOG = 0;
      gr.mZCOG = 0;
      gr.mNpixels = rowBinEdge * colBinEdge;
      gr.mIsGroup = true;
      gr.mFrequency = 0.;
      /// A dummy pattern with all fired pixels in the bounding box is assigned to groups of rare topologies.
      unsigned char dummyPattern[ClusterPattern::kExtendedPatternBytes] = {0};
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
    index = LookUp::groupFinder(rs, cs);
    tmp_GroupMap[index].second += mTopologyFrequency[j].first;
  }

  for (auto&& p : tmp_GroupMap) {
    GroupStruct& group = p.second.first;
    group.mFrequency = ((double)p.second.second) / mTotClusters;
    mDictionary.mVectorOfIDs.push_back(group);
  }

  // Sorting the dictionary
  std::sort(mDictionary.mVectorOfIDs.begin(), mDictionary.mVectorOfIDs.end(), [](const GroupStruct& a, const GroupStruct& b) { return (a.mFrequency > b.mFrequency); });
  // Creating the map for common topologies
  for (int iKey = 0; iKey < mDictionary.getSize(); iKey++) {
    GroupStruct& gr = mDictionary.mVectorOfIDs[iKey];
    if (!gr.mIsGroup) {
      mDictionary.mCommonMap.insert(std::make_pair(gr.mHash, iKey));
      if (gr.mPattern.getUsedBytes() == 1) {
        mDictionary.mSmallTopologiesLUT[(gr.mPattern.getColumnSpan() - 1) * 255 + (int)gr.mPattern.mBitmap[2]] = iKey;
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

std::ostream& operator<<(std::ostream& os, const BuildTopologyDictionary& DB)
{
  for (int i = 0; i < DB.mNCommonTopologies; i++) {
    const unsigned long& hash = DB.mTopologyFrequency[i].second;
    os << "Hash: " << hash << std::endl;
    os << "counts: " << DB.mTopologyMap.find(hash)->second.countsTotal;
    os << " (with bias provided: " << DB.mTopologyMap.find(hash)->second.countsWithBias << ")" << std::endl;
    os << "sigmaX: " << std::sqrt(DB.mMapInfo.find(hash)->second.mXsigma2) << std::endl;
    os << "sigmaZ: " << std::sqrt(DB.mMapInfo.find(hash)->second.mZsigma2) << std::endl;
    os << DB.mTopologyMap.find(hash)->second.topology;
  }
  return os;
}

void BuildTopologyDictionary::printDictionary(const std::string& fname)
{
  std::cout << "Saving the the dictionary in binary format: ";
  std::ofstream out(fname);
  out << mDictionary;
  out.close();
  std::cout << "done!" << std::endl;
}

void BuildTopologyDictionary::printDictionaryBinary(const std::string& fname)
{
  std::cout << "Printing the dictionary: ";
  std::ofstream out(fname);
  mDictionary.writeBinaryFile(fname);
  out.close();
  std::cout << "done!" << std::endl;
}

void BuildTopologyDictionary::saveDictionaryRoot(const std::string& fname)
{
  std::cout << "Saving the the dictionary in a ROOT file: ";
  TFile output(fname.c_str(), "recreate");
  output.WriteObjectAny(&mDictionary, mDictionary.Class(), "TopologyDictionary");
  output.Close();
  std::cout << "done!" << std::endl;
}

} // namespace itsmft
} // namespace o2
