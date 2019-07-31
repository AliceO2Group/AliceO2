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

ClassImp(o2::itsmft::BuildTopologyDictionary)

  namespace o2
{
  namespace itsmft
  {
  constexpr float BuildTopologyDictionary::IgnoreVal;

  BuildTopologyDictionary::BuildTopologyDictionary() : mTotClusters{ 0 } {}

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
      //__________________COG_Deterrmination_____________
      int tempxCOG = 0, tempzCOG = 0, tempFiredPixels = 0, s = 0, ic = 0, ir = 0;
      unsigned char tempChar = 0;
      for (unsigned int i = 2; i < cluster.getUsedBytes() + 2; i++) {
        tempChar = cluster.getByte(i);
        s = 128; // 0b10000000
        while (s > 0) {
          if ((tempChar & s) != 0) {
            tempFiredPixels++;
            tempxCOG += ir;
            tempzCOG += ic;
          }
          ic++;
          s /= 2;
          if ((ir + 1) * ic == (rs * cs)) {
            break;
          }
          if (ic == cs) {
            ic = 0;
            ir++;
          }
        }
        if ((ir + 1) * ic == (rs * cs)) {
          break;
        }
      }
      topInf.mCOGx = (float)tempxCOG / (float)tempFiredPixels;
      topInf.mCOGz = (float)tempzCOG / (float)tempFiredPixels;
      topInf.mNpixels = tempFiredPixels;
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
    mNotInGroups = 0;
    mNumberOfGroups = 0;
    mDictionary.mFinalMap.clear();
    mFrequencyThreshold = thr;
    for (auto& q : mTopologyFrequency) {
      if (((double)q.first) / mTotClusters > thr)
        mNotInGroups++;
      else
        break;
    }
    mNumberOfGroups = mNotInGroups;
  }

  void BuildTopologyDictionary::setNGroups(unsigned int ngr)
  {
    mTopologyFrequency.clear();
    for (auto&& p : mTopologyMap) { // p os pair<ulong,TopoStat>
      mTopologyFrequency.emplace_back(std::make_pair(p.second.countsTotal, p.first));
    }
    std::sort(mTopologyFrequency.begin(), mTopologyFrequency.end(),
              [](const std::pair<unsigned long, unsigned long>& couple1,
                 const std::pair<unsigned long, unsigned long>& couple2) { return (couple1.first > couple2.first); });
    int maxNumberOfGroups = Cluster::kMaxPatternBits / (TopologyDictionary::RowClassSpan * TopologyDictionary::ColClassSpan) * (1 + ceil(log(Cluster::kMaxPatternBits / (TopologyDictionary::RowClassSpan * TopologyDictionary::ColClassSpan))));
    if (ngr < maxNumberOfGroups ||
        ngr > (mTopologyFrequency.size() -
               maxNumberOfGroups)) {
      std::cout << "BuildTopologyDictionary::setNGroups : Invalid number of groups" << std::endl;
      exit(1);
    }
    mNumberOfGroups = mNotInGroups =
      ngr - maxNumberOfGroups;
    mDictionary.mFinalMap.clear();
    mFrequencyThreshold = ((double)mTopologyFrequency[mNotInGroups - 1].first) / mTotClusters;
  }

  void BuildTopologyDictionary::setThresholdCumulative(double cumulative)
  {
    mTopologyFrequency.clear();
    if (cumulative <= 0. || cumulative >= 1.)
      cumulative = 0.99;
    double totFreq = 0.;
    for (auto&& p : mTopologyMap) { // p os pair<ulong,TopoStat>
      mTopologyFrequency.emplace_back(std::make_pair(p.second.countsTotal, p.first));
    }
    std::sort(mTopologyFrequency.begin(), mTopologyFrequency.end(),
              [](const std::pair<unsigned long, unsigned long>& couple1,
                 const std::pair<unsigned long, unsigned long>& couple2) { return (couple1.first > couple2.first); });
    mNotInGroups = 0;
    mNumberOfGroups = 0;
    mDictionary.mFinalMap.clear();
    for (auto& q : mTopologyFrequency) {
      totFreq += ((double)(q.first)) / mTotClusters;
      if (totFreq < cumulative) {
        mNotInGroups++;
      } else
        break;
    }
    mFrequencyThreshold = ((double)(mTopologyFrequency[--mNotInGroups].first)) / mTotClusters;
    while (((double)mTopologyFrequency[mNotInGroups].first) / mTotClusters == mFrequencyThreshold)
      mNotInGroups--;
    mFrequencyThreshold = ((double)mTopologyFrequency[mNotInGroups++].first) / mTotClusters;
    mNumberOfGroups = mNotInGroups;
  }

  void BuildTopologyDictionary::groupRareTopologies()
  {
    std::cout << "groupRareTopologies: mTotClusters: " << mTotClusters << std::endl;
#ifdef _HISTO_
    mHdist =
      TH1F("mHdist", "Groups distribution",
           mNumberOfGroups + TopologyDictionary::NumberOfRareGroups, -0.5,
           mNumberOfGroups + TopologyDictionary::NumberOfRareGroups - 0.5);
    mHdist.GetXaxis()->SetTitle("GroupID");
    mHdist.SetFillColor(kRed);
    mHdist.SetFillStyle(3005);
#endif

    double totFreq = 0.;
    for (int j = 0; j < mNotInGroups; j++) {
#ifdef _HISTO_
      mHdist.Fill(j, mTopologyFrequency[j].first);
#endif
      totFreq += ((double)(mTopologyFrequency[j].first)) / mTotClusters;
      GroupStruct gr;
      gr.mHash = mTopologyFrequency[j].second;
      gr.mFrequency = totFreq;
      // rough estimation for the error considering a8 uniform distribution
      gr.mErrX = std::sqrt(mMapInfo.find(gr.mHash)->second.mXsigma2);
      gr.mErrZ = std::sqrt(mMapInfo.find(gr.mHash)->second.mZsigma2);
      gr.mXCOG = -1 * mMapInfo.find(gr.mHash)->second.mCOGx * o2::itsmft::SegmentationAlpide::PitchRow;
      gr.mZCOG = mMapInfo.find(gr.mHash)->second.mCOGz * o2::itsmft::SegmentationAlpide::PitchCol;
      gr.mNpixels = mMapInfo.find(gr.mHash)->second.mNpixels;
      gr.mPattern = mMapInfo.find(gr.mHash)->second.mPattern;
      gr.mIsGroup = false;
      mDictionary.mVectorOfGroupIDs.push_back(gr);
      mDictionary.mFinalMap.insert(std::make_pair(gr.mHash, j));
      if (gr.mPattern.getUsedBytes() == 1)
        mDictionary.mSmallTopologiesLUT[(gr.mPattern.getColumnSpan() - 1) * 255 + (int)gr.mPattern.mBitmap[2]] = j;
    }
    // groupRareTopologies based on binning over number of rows and columns (TopologyDictionary::NumberOfRowClasses *
    // NumberOfColClasse)

    mNumberOfGroups += TopologyDictionary::NumberOfRareGroups;
    // array of groups
    std::array<GroupStruct, TopologyDictionary::NumberOfRareGroups> GroupArray;
    std::array<unsigned long, TopologyDictionary::NumberOfRareGroups>
      groupCounts{ 0 };
    auto createGroupEntry = [&GroupArray](int rowBinEdge, int colBinEdge, int& index) {
      unsigned long provvHash = 0;
      provvHash = (((unsigned long)(index + 1)) << 32) & 0xffffffff00000000;
      GroupArray[index].mHash = provvHash;
      GroupArray[index].mErrX = (rowBinEdge)*o2::itsmft::SegmentationAlpide::PitchRow / std::sqrt(12);
      GroupArray[index].mErrZ = (colBinEdge)*o2::itsmft::SegmentationAlpide::PitchCol / std::sqrt(12);
      GroupArray[index].mXCOG = 0;
      GroupArray[index].mZCOG = 0;
      GroupArray[index].mNpixels = rowBinEdge * colBinEdge;
      GroupArray[index].mIsGroup = true;
      unsigned char dummyPattern[ClusterPattern::kExtendedPatternBytes] = {
        0
      }; /// A dummy pattern with all fired pixels in the bounding box is assigned to groups of rare topologies.
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
      GroupArray[index].mPattern.setPattern(dummyPattern);
      index++;
      return;
    };
    auto createPlaceHolder = [&GroupArray](int& index) { // Filling the dictionary with place-holders for entries of impossible groups due to pattern size.
      GroupArray[index].mHash = 0;
      GroupArray[index].mErrX = 0;
      GroupArray[index].mErrZ = 0;
      GroupArray[index].mXCOG = 0;
      GroupArray[index].mZCOG = 0;
      GroupArray[index].mNpixels = 0;
      GroupArray[index].mIsGroup = true;
      unsigned char dummyPattern[ClusterPattern::kExtendedPatternBytes] = {
        0
      };
      dummyPattern[0] = (unsigned char)TopologyDictionary::RowClassSpan;
      dummyPattern[1] = (unsigned char)TopologyDictionary::ColClassSpan;
      GroupArray[index].mPattern.setPattern(dummyPattern);
      index++;
      return;
    };

    int grNum = 0;
    for (int iRow = 0; iRow < TopologyDictionary::MaxNumberOfClasses; iRow++) {
      for (int iCol = 0; iCol < TopologyDictionary::MaxNumberOfClasses; iCol++) {
        if ((iRow + 1) * (iCol + 1) <= TopologyDictionary::MaxNumberOfClasses) {
          createGroupEntry((iRow + 1) * TopologyDictionary::RowClassSpan, (iCol + 1) * TopologyDictionary::ColClassSpan, grNum);
        } else
          createPlaceHolder(grNum);
      }
    }

    int rs;
    int cs;
    int index;

    for (unsigned int j = (unsigned int)mNotInGroups; j < mTopologyFrequency.size(); j++) {
      unsigned long hash1 = mTopologyFrequency[j].second;
      rs = mTopologyMap.find(hash1)->second.topology.getRowSpan();
      cs = mTopologyMap.find(hash1)->second.topology.getColumnSpan();
      index = LookUp::groupFinder(rs, cs);
      groupCounts[index] += mTopologyFrequency[j].first;
    }

    for (int i = 0; i < TopologyDictionary::MaxNumberOfClasses * TopologyDictionary::MaxNumberOfClasses; i++) {
      totFreq += ((double)groupCounts[i]) / mTotClusters;
      GroupArray[i].mFrequency = totFreq;
#ifdef _HISTO_
      mHdist.Fill(mNotInGroups + i, groupCounts[i]);
#endif
      mDictionary.mVectorOfGroupIDs.push_back(GroupArray[i]);
    }
#ifdef _HISTO_
    mHdist.Scale(1. / mHdist.Integral());
#endif

    // Filling Look-up table for small topologies
  }

  std::ostream& operator<<(std::ostream& os, const BuildTopologyDictionary& DB)
  {
    for (int i = 0; i < DB.mNotInGroups; i++) {
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

  void BuildTopologyDictionary::printDictionary(std::string fname)
  {
    std::ofstream out(fname);
    out << mDictionary;
    out.close();
  }

  void BuildTopologyDictionary::printDictionaryBinary(std::string fname)
  {
    std::ofstream out(fname);
    mDictionary.WriteBinaryFile(fname);
    out.close();
  }

  void BuildTopologyDictionary::saveDictionaryRoot(const char* filename)
  {
    TFile output(filename, "recreate");
    output.WriteObjectAny(&mDictionary, mDictionary.Class(), "TopologyDictionary");
    output.Close();
  }

  } // namespace itsmft
}
