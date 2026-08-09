// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Clusterer.h
/// \brief Definition of the TRK cluster finder

#ifndef ALICEO2_TRK_CLUSTERER_H
#define ALICEO2_TRK_CLUSTERER_H

// uncomment to allow diagonal clusters, e.g. |* |
//                                            | *|
#define _ALLOW_DIAGONAL_TRK_CLUSTERS_

#include "DataFormatsTRKFT3/Digit.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsTRKFT3/Cluster.h"
#include "DataFormatsTRKFT3/ROFRecord.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TRKBase/Specs.h"
#include "MathUtils/Cartesian.h"
#include <gsl/span>
#include <vector>
#include <array>
#include <memory>
#include <cstring>
#include <utility>

namespace o2::trk
{

class GeometryTGeo;

template <int DetID>
class Clusterer
{
  static_assert(DetID == o2::detectors::DetID::TRK || DetID == o2::detectors::DetID::FT3, "only TRK and FT3 clusterers are supported");

 public:
  static constexpr int MaxLabels = 10;
  static constexpr int MaxHugeClusWarn = 5;

  using Digit = o2::trkft3::Digit;
  using DigROFRecord = o2::trkft3::ROFRecord;
  using ClusterType = o2::trkft3::Cluster<DetID>;
  using ClusterTruth = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  using ConstDigitTruth = o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel>;
  using Label = o2::MCCompLabel;

  //----------------------------------------------
  struct BBox {
    uint16_t chipID = 0xffff;
    uint16_t rowMin = 0xffff, colMin = 0xffff;
    uint16_t rowMax = 0, colMax = 0;
    explicit BBox(uint16_t c) : chipID(c) {}
    bool isInside(uint16_t r, uint16_t c) const { return r >= rowMin && r <= rowMax && c >= colMin && c <= colMax; }
    uint16_t rowSpan() const { return rowMax - rowMin + 1; }
    uint16_t colSpan() const { return colMax - colMin + 1; }
    bool isAcceptableSize() const
    {
      return rowSpan() <= o2::itsmft::ClusterPattern::MaxRowSpan &&
             colSpan() <= o2::itsmft::ClusterPattern::MaxColSpan;
    }
    void adjust(uint16_t r, uint16_t c)
    {
      if (r < rowMin) {
        rowMin = r;
      }
      if (r > rowMax) {
        rowMax = r;
      }
      if (c < colMin) {
        colMin = c;
      }
      if (c > colMax) {
        colMax = c;
      }
    }
  };

  //----------------------------------------------
  struct ClustererThread {
    Clusterer<DetID>* parent = nullptr;
    // column buffers (pre-cluster state); extra sentinel entries at [0] and [size-1]
    int* column1 = nullptr;
    int* column2 = nullptr;
    int* curr = nullptr;                               ///< current column pre-cluster indices
    int* prev = nullptr;                               ///< previous column pre-cluster indices
    int size = constants::moduleMLOT::chip::nRows + 2; ///< reallocated per chip in initChip

    // pixels[i] = {next_in_chain, global_digit_index}
    std::vector<std::pair<int, uint32_t>> pixels;
    std::vector<int> preClusterHeads;
    std::vector<int> preClusterIndices;
    uint16_t currCol = 0xffff;
    bool noLeftCol = true;

    std::array<Label, MaxLabels> labelsBuff;               ///< MC label buffer for one cluster
    std::vector<std::pair<uint16_t, uint16_t>> pixArrBuff; ///< (row,col) pixel buffer for pattern

    // per-thread output (accumulated, then merged back by caller)
    std::vector<ClusterType> clusters;
    std::vector<unsigned char> patterns;
    ClusterTruth labels;

    ///< reset column buffer
    void resetColumn(int* buff) const { std::memset(buff, -1, sizeof(int) * (size - 2)); }
    ///< swap current and previous column buffers
    void swapColumnBuffers() { std::swap(prev, curr); }

    ///< append pixel ip to the pre-cluster headed at preClusterIndex
    void expandPreCluster(uint32_t ip, uint16_t row, int preClusterIndex)
    {
      auto& firstIndex = preClusterHeads[preClusterIndices[preClusterIndex]];
      pixels.emplace_back(firstIndex, ip);
      firstIndex = pixels.size() - 1;
      curr[row] = preClusterIndex;
    }

    ///< start a new pre-cluster with pixel ip at given row
    void addNewPreCluster(uint32_t ip, uint16_t row)
    {
      preClusterHeads.push_back(pixels.size());
      pixels.emplace_back(-1, ip);
      int lastIndex = preClusterIndices.size();
      preClusterIndices.push_back(lastIndex);
      curr[row] = lastIndex;
    }

    void fetchMCLabels(uint32_t digID, const ConstDigitTruth* labelsDig, int& nfilled);
    void initChip(gsl::span<const Digit> digits, uint32_t first, GeometryTGeo* geom);
    void updateChip(gsl::span<const Digit> digits, uint32_t ip);
    void finishChip(gsl::span<const Digit> digits,
                    const ConstDigitTruth* labelsDigPtr, ClusterTruth* labelsClusPtr,
                    GeometryTGeo* geom);
    void finishChipSingleHitFast(gsl::span<const Digit> digits, uint32_t hit,
                                 const ConstDigitTruth* labelsDigPtr, ClusterTruth* labelsClusPtr,
                                 GeometryTGeo* geom);
    void processChip(gsl::span<const Digit> digits, int chipFirst, int chipN,
                     std::vector<ClusterType>* clustersOut, std::vector<unsigned char>* patternsOut,
                     const ConstDigitTruth* labelsDigPtr, ClusterTruth* labelsClusPtr,
                     GeometryTGeo* geom);
    void streamCluster(const BBox& bbox, const std::vector<std::pair<uint16_t, uint16_t>>& pixbuf,
                       uint32_t totalCharge, bool doLabels, int nlab,
                       uint16_t chipID, int subDetID, int layer);

    ~ClustererThread()
    {
      delete[] column1;
      delete[] column2;
    }
    explicit ClustererThread(Clusterer<DetID>* par = nullptr) : parent(par) {}
    ClustererThread(const ClustererThread&) = delete;
    ClustererThread& operator=(const ClustererThread&) = delete;
  };
  //----------------------------------------------

  virtual void process(gsl::span<const Digit> digits,
                       gsl::span<const DigROFRecord> digitROFs,
                       std::vector<ClusterType>& clusters,
                       std::vector<unsigned char>& patterns,
                       std::vector<o2::trkft3::ROFRecord>& clusterROFs,
                       const ConstDigitTruth* digitLabels = nullptr,
                       ClusterTruth* clusterLabels = nullptr);

  static o2::math_utils::Point3D<float> getClusterLocalCoordinates(const ClusterType& cluster, const uint8_t* patt,
                                                                   float yPlaneMLOT = 0.f) noexcept;

 protected:
  int mNHugeClus = 0;
  std::unique_ptr<ClustererThread> mThread;
  std::vector<int> mSortIdx; ///< reusable per-ROF sort buffer
};

using TRKClusterer = Clusterer<o2::detectors::DetID::TRK>;
using FT3Clusterer = Clusterer<o2::detectors::DetID::FT3>;

} // namespace o2::trk

#endif
