// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCClusterStatistics.cxx
/// \author David Rohr

#include "GPUTPCClusterStatistics.h"
#include "GPULogging.h"
#include "GPUO2DataTypes.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <queue>

using namespace GPUCA_NAMESPACE::gpu;

// Small helper to compute Huffman probabilities
namespace
{
typedef std::vector<bool> HuffCode;
typedef std::map<unsigned int, HuffCode> HuffCodeMap;

class INode
{
 public:
  const double f;

  virtual ~INode() = default;

 protected:
  INode(double v) : f(v) {}
};

class InternalNode : public INode
{
 public:
  INode* const left;
  INode* const right;

  InternalNode(INode* c0, INode* c1) : INode(c0->f + c1->f), left(c0), right(c1) {}
  ~InternalNode() override
  {
    delete left;
    delete right;
  }
};

class LeafNode : public INode
{
 public:
  const unsigned int c;

  LeafNode(double v, unsigned int w) : INode(v), c(w) {}
};

struct NodeCmp {
  bool operator()(const INode* lhs, const INode* rhs) const { return lhs->f > rhs->f; }
};

INode* BuildTree(const double* frequencies, unsigned int UniqueSymbols)
{
  std::priority_queue<INode*, std::vector<INode*>, NodeCmp> trees;

  for (unsigned int i = 0; i < UniqueSymbols; ++i) {
    if (frequencies[i] != 0) {
      trees.push(new LeafNode(frequencies[i], i));
    }
  }
  while (trees.size() > 1) {
    INode* childR = trees.top();
    trees.pop();

    INode* childL = trees.top();
    trees.pop();

    INode* parent = new InternalNode(childR, childL);
    trees.push(parent);
  }
  return trees.top();
}

void GenerateCodes(const INode* node, const HuffCode& prefix, HuffCodeMap& outCodes)
{
  if (const LeafNode* lf = dynamic_cast<const LeafNode*>(node)) {
    outCodes[lf->c] = prefix;
  } else if (const InternalNode* in = dynamic_cast<const InternalNode*>(node)) {
    HuffCode leftPrefix = prefix;
    leftPrefix.push_back(false);
    GenerateCodes(in->left, leftPrefix, outCodes);

    HuffCode rightPrefix = prefix;
    rightPrefix.push_back(true);
    GenerateCodes(in->right, rightPrefix, outCodes);
  }
}
} // namespace

void GPUTPCClusterStatistics::RunStatistics(const o2::tpc::ClusterNativeAccess* clustersNative, const o2::tpc::CompressedClusters* clustersCompressed, const GPUParam& param)
{
  bool decodingError = false;
  o2::tpc::ClusterNativeAccess clustersNativeDecoded;
  std::vector<o2::tpc::ClusterNative> clusterBuffer;
  mDecoder.decompress(clustersCompressed, clustersNativeDecoded, clusterBuffer, param);
  std::vector<o2::tpc::ClusterNative> tmpClusters;
  if (param.rec.tpcRejectionMode == GPUSettings::RejectionNone) { // verification does not make sense if we reject clusters during compression
    for (unsigned int i = 0; i < NSLICES; i++) {
      for (unsigned int j = 0; j < GPUCA_ROW_COUNT; j++) {
        if (clustersNative->nClusters[i][j] != clustersNativeDecoded.nClusters[i][j]) {
          GPUError("Number of clusters mismatch slice %u row %u: expected %d v.s. decoded %d", i, j, clustersNative->nClusters[i][j], clustersNativeDecoded.nClusters[i][j]);
          decodingError = true;
          continue;
        }
        tmpClusters.resize(clustersNative->nClusters[i][j]);
        for (unsigned int k = 0; k < clustersNative->nClusters[i][j]; k++) {
          tmpClusters[k] = clustersNative->clusters[i][j][k];
          if (param.rec.tpcCompressionModes & GPUSettings::CompressionTruncate) {
            GPUTPCCompression::truncateSignificantBitsCharge(tmpClusters[k].qMax, param);
            GPUTPCCompression::truncateSignificantBitsCharge(tmpClusters[k].qTot, param);
            GPUTPCCompression::truncateSignificantBitsWidth(tmpClusters[k].sigmaPadPacked, param);
            GPUTPCCompression::truncateSignificantBitsWidth(tmpClusters[k].sigmaTimePacked, param);
          }
        }
        std::sort(tmpClusters.begin(), tmpClusters.end());
        for (unsigned int k = 0; k < clustersNative->nClusters[i][j]; k++) {
          const o2::tpc::ClusterNative& c1 = tmpClusters[k];
          const o2::tpc::ClusterNative& c2 = clustersNativeDecoded.clusters[i][j][k];
          if (c1.timeFlagsPacked != c2.timeFlagsPacked || c1.padPacked != c2.padPacked || c1.sigmaTimePacked != c2.sigmaTimePacked || c1.sigmaPadPacked != c2.sigmaPadPacked || c1.qMax != c2.qMax || c1.qTot != c2.qTot) {
            GPUWarning("Cluster mismatch: slice %2u row %3u hit %5u: %6d %3d %4d %3d %3d %4d %4d", i, j, k, (int)c1.getTimePacked(), (int)c1.getFlags(), (int)c1.padPacked, (int)c1.sigmaTimePacked, (int)c1.sigmaPadPacked, (int)c1.qMax, (int)c1.qTot);
            GPUWarning("%45s %6d %3d %4d %3d %3d %4d %4d", "", (int)c2.getTimePacked(), (int)c2.getFlags(), (int)c2.padPacked, (int)c2.sigmaTimePacked, (int)c2.sigmaPadPacked, (int)c2.qMax, (int)c2.qTot);
            decodingError = true;
          }
        }
      }
    }
    if (decodingError) {
      mDecodingError = true;
    } else {
      GPUInfo("Cluster decoding verification: PASSED");
    }
  }

  FillStatistic(mPqTotA, clustersCompressed->qTotA, clustersCompressed->nAttachedClusters);
  FillStatistic(mPqMaxA, clustersCompressed->qMaxA, clustersCompressed->nAttachedClusters);
  FillStatistic(mPflagsA, clustersCompressed->flagsA, clustersCompressed->nAttachedClusters);
  FillStatistic(mProwDiffA, clustersCompressed->rowDiffA, clustersCompressed->nAttachedClustersReduced);
  FillStatistic(mPsliceLegDiffA, clustersCompressed->sliceLegDiffA, clustersCompressed->nAttachedClustersReduced);
  FillStatistic(mPpadResA, clustersCompressed->padResA, clustersCompressed->nAttachedClustersReduced);
  FillStatistic(mPtimeResA, clustersCompressed->timeResA, clustersCompressed->nAttachedClustersReduced);
  FillStatistic(mPsigmaPadA, clustersCompressed->sigmaPadA, clustersCompressed->nAttachedClusters);
  FillStatistic(mPsigmaTimeA, clustersCompressed->sigmaTimeA, clustersCompressed->nAttachedClusters);
  FillStatistic(mPqPtA, clustersCompressed->qPtA, clustersCompressed->nTracks);
  FillStatistic(mProwA, clustersCompressed->rowA, clustersCompressed->nTracks);
  FillStatistic(mPsliceA, clustersCompressed->sliceA, clustersCompressed->nTracks);
  FillStatistic(mPtimeA, clustersCompressed->timeA, clustersCompressed->nTracks);
  FillStatistic(mPpadA, clustersCompressed->padA, clustersCompressed->nTracks);
  FillStatistic(mPqTotU, clustersCompressed->qTotU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPqMaxU, clustersCompressed->qMaxU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPflagsU, clustersCompressed->flagsU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPpadDiffU, clustersCompressed->padDiffU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPtimeDiffU, clustersCompressed->timeDiffU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPsigmaPadU, clustersCompressed->sigmaPadU, clustersCompressed->nUnattachedClusters);
  FillStatistic(mPsigmaTimeU, clustersCompressed->sigmaTimeU, clustersCompressed->nUnattachedClusters);
  FillStatistic<short unsigned int, 1>(mPnTrackClusters, clustersCompressed->nTrackClusters, clustersCompressed->nTracks);
  FillStatistic<unsigned int, 1>(mPnSliceRowClusters, clustersCompressed->nSliceRowClusters, clustersCompressed->nSliceRows);
  FillStatisticCombined(mPsigmaA, clustersCompressed->sigmaPadA, clustersCompressed->sigmaTimeA, clustersCompressed->nAttachedClusters, 1 << 8);
  FillStatisticCombined(mPsigmaU, clustersCompressed->sigmaPadU, clustersCompressed->sigmaTimeU, clustersCompressed->nUnattachedClusters, 1 << 8);
  FillStatisticCombined(mProwSliceA, clustersCompressed->rowDiffA, clustersCompressed->sliceLegDiffA, clustersCompressed->nAttachedClustersReduced, GPUCA_ROW_COUNT);
  mNTotalClusters += clustersCompressed->nAttachedClusters + clustersCompressed->nUnattachedClusters;
}

void GPUTPCClusterStatistics::Finish()
{
  if (mDecodingError) {
    GPUError("-----------------------------------------\nERROR - INCORRECT CLUSTER DECODING!\n-----------------------------------------");
  }
  if (mNTotalClusters == 0) {
    return;
  }

  GPUInfo("\nRunning cluster compression entropy statistics");
  Analyze(mPqTotA, "qTot Attached");
  Analyze(mPqMaxA, "qMax Attached");
  Analyze(mPflagsA, "flags Attached");
  double eRowSlice = Analyze(mProwDiffA, "rowDiff Attached", false);
  eRowSlice += Analyze(mPsliceLegDiffA, "sliceDiff Attached", false);
  Analyze(mPpadResA, "padRes Attached");
  Analyze(mPtimeResA, "timeRes Attached");
  double eSigma = Analyze(mPsigmaPadA, "sigmaPad Attached", false);
  eSigma += Analyze(mPsigmaTimeA, "sigmaTime Attached", false);
  Analyze(mPqPtA, "qPt Attached");
  Analyze(mProwA, "row Attached");
  Analyze(mPsliceA, "slice Attached");
  Analyze(mPtimeA, "time Attached");
  Analyze(mPpadA, "pad Attached");
  Analyze(mPqTotU, "qTot Unattached");
  Analyze(mPqMaxU, "qMax Unattached");
  Analyze(mPflagsU, "flags Unattached");
  Analyze(mPpadDiffU, "padDiff Unattached");
  Analyze(mPtimeDiffU, "timeDiff Unattached");
  eSigma += Analyze(mPsigmaPadU, "sigmaPad Unattached", false);
  eSigma += Analyze(mPsigmaTimeU, "sigmaTime Unattached", false);
  Analyze(mPnTrackClusters, "nClusters in Track");
  Analyze(mPnSliceRowClusters, "nClusters in Row");
  double eSigmaCombined = Analyze(mPsigmaA, "combined sigma Attached");
  eSigmaCombined += Analyze(mPsigmaU, "combined sigma Unattached");
  double eRowSliceCombined = Analyze(mProwSliceA, "combined row/slice Attached");

  GPUInfo("Combined Row/Slice: %6.4f --> %6.4f (%6.4f%%)", eRowSlice, eRowSliceCombined, 100. * (eRowSlice - eRowSliceCombined) / eRowSlice);
  GPUInfo("Combined Sigma: %6.4f --> %6.4f (%6.4f%%)", eSigma, eSigmaCombined, 100. * (eSigma - eSigmaCombined) / eSigma);

  printf("\nConbined Entropy: %7.4f   (Size %'13.0f, %'lld cluster)\nCombined Huffman: %7.4f   (Size %'13.0f, %f%%)\n\n", mEntropy / mNTotalClusters, mEntropy, (long long int)mNTotalClusters, mHuffman / mNTotalClusters, mHuffman, 100. * (mHuffman - mEntropy) / mHuffman);
}

float GPUTPCClusterStatistics::Analyze(std::vector<int>& p, const char* name, bool count)
{
  double entropy = 0.;
  double huffmanSize = 0;

  std::vector<double> prob(p.size());
  double log2 = log(2.);
  size_t total = 0;
  for (unsigned int i = 0; i < p.size(); i++) {
    total += p[i];
  }
  for (unsigned int i = 0; i < prob.size(); i++) {
    if (total && p[i]) {
      prob[i] = (double)p[i] / total;
      double I = -log(prob[i]) / log2;
      double H = I * prob[i];

      entropy += H;
    }
  }

  INode* root = BuildTree(prob.data(), prob.size());

  HuffCodeMap codes;
  GenerateCodes(root, HuffCode(), codes);
  delete root;

  for (HuffCodeMap::const_iterator it = codes.begin(); it != codes.end(); ++it) {
    huffmanSize += it->second.size() * prob[it->first];
  }

  if (count) {
    mEntropy += entropy * total;
    mHuffman += huffmanSize * total;
  }
  GPUInfo("Size: %30s: Entropy %7.4f Huffman %7.4f", name, entropy, huffmanSize);
  return entropy;
}

template <class T, int I>
void GPUTPCClusterStatistics::FillStatistic(std::vector<int>& p, const T* ptr, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    unsigned int val = ptr[i];
    if (I && p.size() <= val + 1) {
      p.resize(val + 1);
    }
    p[val]++;
  }
}

template <class T, class S, int I>
void GPUTPCClusterStatistics::FillStatisticCombined(std::vector<int>& p, const T* ptr1, const S* ptr2, size_t n, int max1)
{
  for (size_t i = 0; i < n; i++) {
    unsigned int val = ptr1[i] + ptr2[i] * max1;
    if (I && p.size() < val + 1) {
      p.resize(val + 1);
    }
    p[val]++;
  }
}
