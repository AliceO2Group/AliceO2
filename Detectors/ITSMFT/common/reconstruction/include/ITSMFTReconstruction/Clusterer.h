// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.h
/// \brief Definition of the ITS cluster finder
#ifndef ALICEO2_ITS_CLUSTERER_H
#define ALICEO2_ITS_CLUSTERER_H

#include <utility>
#include <vector>
#include "ITSMFTBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/Cluster.h"
#include "ITSMFTReconstruction/PixelReader.h"

#include "Rtypes.h"

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace ITSMFT
{
class Clusterer
{
  using PixelReader = o2::ITSMFT::PixelReader;
  using PixelData = o2::ITSMFT::PixelReader::PixelData;
  using ChipPixelData = o2::ITSMFT::PixelReader::ChipPixelData;
  using Cluster = o2::ITSMFT::Cluster;
  using Label = o2::MCCompLabel;

 public:
  Clusterer();
  ~Clusterer() = default;

  Clusterer(const Clusterer&) = delete;
  Clusterer& operator=(const Clusterer&) = delete;

  void process(PixelReader& r, std::vector<Cluster>& clusters);

  // provide the common ITSMFT::GeometryTGeo to access matrices
  void setGeometry(const o2::ITSMFT::GeometryTGeo* gm) { mGeometry = gm; }
  void setMCTruthContainer(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* truth) { mClsLabels = truth; }

 private:
  enum { kMaxRow = 650 }; // Anything larger than the real number of rows (512 for ALPIDE)
  void initChip();
  void updateChip(int ip);
  void finishChip(std::vector<Cluster>& clusters);
  void fetchMCLabels(const PixelData* pix, std::array<Label, Cluster::maxLabels>& labels, int& nfilled) const;

  ChipPixelData mChipData; ///< single chip data provided by the reader

  Int_t mColumn1[kMaxRow + 2];
  Int_t mColumn2[kMaxRow + 2];
  Int_t *mCurr, *mPrev;

  using NextIndex = Int_t;
  std::vector<std::pair<NextIndex, const PixelData*>> mPixels;

  using FirstIndex = Int_t;
  std::vector<FirstIndex> mPreClusterHeads;

  std::vector<Int_t> mPreClusterIndices;

  UShort_t mCol = 0xffff; ///< Column being processed

  const o2::ITSMFT::GeometryTGeo* mGeometry = nullptr;                      ///< ITS OR MFT upgrade geometry
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mClsLabels = nullptr; // Cluster MC labels
};

} // namespace ITSMFT
} // namespace o2
#endif /* ALICEO2_ITS_CLUSTERER_H */
