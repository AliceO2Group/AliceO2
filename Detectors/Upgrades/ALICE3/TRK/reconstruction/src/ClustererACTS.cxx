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

///
/// \file ClustererACTS.cxx
/// \brief Implementation of the TRK cluster finder with the ACTS
/// \author Nicolò Jacazio, Università del Piemonte Orientale (IT)
/// \since 2026-03-01
///

#include "TRKReconstruction/ClustererACTS.h"
#include "TRKBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include <Acts/Clusterization/Clusterization.hpp>

#include <algorithm>
#include <array>
#include <numeric>

using namespace o2::trk;

// Data formats for ACTS interface
struct Cell2D {
  Cell2D(int rowv, int colv, uint32_t digIdx = 0) : row(rowv), col(colv), digitIdx(digIdx) {}
  int row, col;
  uint32_t digitIdx; ///< Index of the original digit (for MC label retrieval)
  Acts::Ccl::Label label{Acts::Ccl::NO_LABEL};
};

int getCellRow(const Cell2D& cell)
{
  return cell.row;
}

int getCellColumn(const Cell2D& cell)
{
  return cell.col;
}

bool operator==(const Cell2D& left, const Cell2D& right)
{
  return left.row == right.row && left.col == right.col;
}

bool cellComp(const Cell2D& left, const Cell2D& right)
{
  return (left.row == right.row) ? left.col < right.col : left.row < right.row;
}

struct Cluster2D {
  std::vector<Cell2D> cells;
  std::size_t hash{0};
};

void clusterAddCell(Cluster2D& cl, const Cell2D& cell)
{
  cl.cells.push_back(cell);
}

void hash(Cluster2D& cl)
{
  std::ranges::sort(cl.cells, cellComp);
  cl.hash = 0;
  // for (const Cell2D& c : cl.cells) {
  //   boost::hash_combine(cl.hash, c.col);
  // }
}

bool clHashComp(const Cluster2D& left, const Cluster2D& right)
{
  return left.hash < right.hash;
}

template <typename RNG>
void genclusterw(int x, int y, int x0, int y0, int x1, int y1,
                 std::vector<Cell2D>& cells, RNG& rng, double startp = 0.5,
                 double decayp = 0.9)
{
  std::vector<Cell2D> add;

  auto maybe_add = [&](int x_, int y_) {
    Cell2D c(x_, y_);
    // if (std::uniform_real_distribution<double>()(rng) < startp &&
    //     !rangeContainsValue(cells, c)) {
    //   cells.push_back(c);
    //   add.push_back(c);
    // }
  };

  // NORTH
  if (y < y1) {
    maybe_add(x, y + 1);
  }
  // NORTHEAST
  if (x < x1 && y < y1) {
    maybe_add(x + 1, y + 1);
  }
  // EAST
  if (x < x1) {
    maybe_add(x + 1, y);
  }
  // SOUTHEAST
  if (x < x1 && y > y0) {
    maybe_add(x + 1, y - 1);
  }
  // SOUTH
  if (y > y0) {
    maybe_add(x, y - 1);
  }
  // SOUTHWEST
  if (x > x0 && y > y0) {
    maybe_add(x - 1, y - 1);
  }
  // WEST
  if (x > x0) {
    maybe_add(x - 1, y);
  }
  // NORTHWEST
  if (x > x0 && y < y1) {
    maybe_add(x - 1, y + 1);
  }

  for (Cell2D& c : add) {
    genclusterw(c.row, c.col, x0, y0, x1, y1, cells, rng, startp * decayp,
                decayp);
  }
}

template <typename RNG>
Cluster2D gencluster(int x0, int y0, int x1, int y1, RNG& rng,
                     double startp = 0.5, double decayp = 0.9)
{
  int x0_ = x0 + 1;
  int x1_ = x1 - 1;
  int y0_ = y0 + 1;
  int y1_ = y1 - 1;

  int x = std::uniform_int_distribution<std::int32_t>(x0_, x1_)(rng);
  int y = std::uniform_int_distribution<std::int32_t>(y0_, y1_)(rng);

  std::vector<Cell2D> cells = {Cell2D(x, y)};
  genclusterw(x, y, x0_, y0_, x1_, y1_, cells, rng, startp, decayp);

  Cluster2D cl;
  cl.cells = std::move(cells);

  return cl;
}

//__________________________________________________
void ClustererACTS::process(gsl::span<const Digit> digits,
                            gsl::span<const DigROFRecord> digitROFs,
                            std::vector<o2::trk::Cluster>& clusters,
                            std::vector<unsigned char>& patterns,
                            std::vector<o2::trk::ROFRecord>& clusterROFs,
                            const ConstDigitTruth* digitLabels,
                            ClusterTruth* clusterLabels,
                            gsl::span<const DigMC2ROFRecord> digMC2ROFs,
                            std::vector<o2::trk::MC2ROFRecord>* clusterMC2ROFs)
{
  if (!mThread) {
    mThread = std::make_unique<ClustererThread>(this);
  }

  auto* geom = o2::trk::GeometryTGeo::Instance();

  for (size_t iROF = 0; iROF < digitROFs.size(); ++iROF) {
    const auto& inROF = digitROFs[iROF];
    const auto outFirst = static_cast<int>(clusters.size());
    const int first = inROF.getFirstEntry();
    const int nEntries = inROF.getNEntries();

    if (nEntries == 0) {
      clusterROFs.emplace_back(inROF.getBCData(), inROF.getROFrame(), outFirst, 0);
      continue;
    }

    // Sort digit indices within this ROF by (chipID, col, row) so we can process
    // chip by chip, column by column -- the same ordering the ALPIDE scanner expects.
    mSortIdx.resize(nEntries);
    std::iota(mSortIdx.begin(), mSortIdx.end(), first);
    std::sort(mSortIdx.begin(), mSortIdx.end(), [&digits](int a, int b) {
      const auto& da = digits[a];
      const auto& db = digits[b];
      if (da.getChipIndex() != db.getChipIndex()) {
        return da.getChipIndex() < db.getChipIndex();
      }
      if (da.getColumn() != db.getColumn()) {
        return da.getColumn() < db.getColumn();
      }
      return da.getRow() < db.getRow();
    });

    // Type aliases for ACTS clustering
    using Cell = Cell2D;
    using CellCollection = std::vector<Cell>;
    using Cluster = Cluster2D;
    using ClusterCollection = std::vector<Cluster>;
    static constexpr int GridDim = 2; ///< Dimensionality of the clustering grid (2D for pixel detectors)

    CellCollection cells;            // Input collection of cells (pixels) to be clustered
    Acts::Ccl::ClusteringData data;  // Internal data structure used by ACTS clustering algorithm
    ClusterCollection clsCollection; // Output collection of clusters found by the algorithm

    // Process one chip at a time
    int sliceStart = 0;
    while (sliceStart < nEntries) {
      const int chipFirst = sliceStart;
      const uint16_t chipID = digits[mSortIdx[sliceStart]].getChipIndex();
      while (sliceStart < nEntries && digits[mSortIdx[sliceStart]].getChipIndex() == chipID) {
        ++sliceStart;
      }
      const int chipN = sliceStart - chipFirst;

      // Fill cells from digits for this chip
      cells.clear();
      data.clear();
      clsCollection.clear();
      cells.reserve(chipN);
      for (int i = chipFirst; i < chipFirst + chipN; ++i) {
        const auto& digit = digits[mSortIdx[i]];
        cells.emplace_back(digit.getRow(), digit.getColumn(), mSortIdx[i]);
      }

      LOG(debug) << "Clustering with ACTS on chip " << chipID << " " << cells.size() << " digits";
      Acts::Ccl::createClusters<CellCollection, ClusterCollection, GridDim>(data,
                                                                            cells,
                                                                            clsCollection,
                                                                            Acts::Ccl::DefaultConnect<Cell, GridDim>(false));

      LOG(debug) << "    found " << clsCollection.size() << " clusters";

      // Convert ACTS clusters to O2 clusters
      for (const auto& actsCluster : clsCollection) {
        if (actsCluster.cells.empty()) {
          continue;
        }

        // Calculate bounding box
        uint16_t rowMin = static_cast<uint16_t>(actsCluster.cells[0].row);
        uint16_t rowMax = rowMin;
        uint16_t colMin = static_cast<uint16_t>(actsCluster.cells[0].col);
        uint16_t colMax = colMin;

        for (const auto& cell : actsCluster.cells) {
          rowMin = std::min(rowMin, static_cast<uint16_t>(cell.row));
          rowMax = std::max(rowMax, static_cast<uint16_t>(cell.row));
          colMin = std::min(colMin, static_cast<uint16_t>(cell.col));
          colMax = std::max(colMax, static_cast<uint16_t>(cell.col));
        }

        const uint16_t rowSpan = rowMax - rowMin + 1;
        const uint16_t colSpan = colMax - colMin + 1;

        // Check if cluster needs splitting (too large for pattern encoding)
        const bool isHuge = rowSpan > o2::itsmft::ClusterPattern::MaxRowSpan ||
                            colSpan > o2::itsmft::ClusterPattern::MaxColSpan;

        if (isHuge) {
          // Split huge cluster into MaxRowSpan x MaxColSpan tiles
          LOG(warning) << "Splitting huge TRK cluster: chipID " << chipID
                       << ", rows " << rowMin << ":" << rowMax
                       << " cols " << colMin << ":" << colMax;

          for (uint16_t tileColMin = colMin; tileColMin <= colMax;
               tileColMin = static_cast<uint16_t>(tileColMin + o2::itsmft::ClusterPattern::MaxColSpan)) {
            uint16_t tileColMax = std::min(colMax, static_cast<uint16_t>(tileColMin + o2::itsmft::ClusterPattern::MaxColSpan - 1));

            for (uint16_t tileRowMin = rowMin; tileRowMin <= rowMax;
                 tileRowMin = static_cast<uint16_t>(tileRowMin + o2::itsmft::ClusterPattern::MaxRowSpan)) {
              uint16_t tileRowMax = std::min(rowMax, static_cast<uint16_t>(tileRowMin + o2::itsmft::ClusterPattern::MaxRowSpan - 1));

              // Collect cells in this tile
              std::vector<std::pair<uint16_t, uint16_t>> tileCells;
              for (const auto& cell : actsCluster.cells) {
                uint16_t r = static_cast<uint16_t>(cell.row);
                uint16_t c = static_cast<uint16_t>(cell.col);
                if (r >= tileRowMin && r <= tileRowMax && c >= tileColMin && c <= tileColMax) {
                  tileCells.emplace_back(r, c);
                }
              }

              if (tileCells.empty()) {
                continue;
              }

              uint16_t tileRowSpan = tileRowMax - tileRowMin + 1;
              uint16_t tileColSpan = tileColMax - tileColMin + 1;

              // Encode pattern for this tile
              std::array<unsigned char, o2::itsmft::ClusterPattern::MaxPatternBytes> patt{};
              for (const auto& [r, c] : tileCells) {
                uint32_t ir = r - tileRowMin;
                uint32_t ic = c - tileColMin;
                int nbit = ir * tileColSpan + ic;
                patt[nbit >> 3] |= (0x1 << (7 - (nbit % 8)));
              }
              patterns.emplace_back(static_cast<unsigned char>(tileRowSpan));
              patterns.emplace_back(static_cast<unsigned char>(tileColSpan));
              const int nBytes = (tileRowSpan * tileColSpan + 7) / 8;
              patterns.insert(patterns.end(), patt.begin(), patt.begin() + nBytes);

              // Handle MC labels for this tile
              if (clusterLabels && digitLabels) {
                const auto clsIdx = static_cast<uint32_t>(clusters.size());
                for (const auto& cell : actsCluster.cells) {
                  uint16_t r = static_cast<uint16_t>(cell.row);
                  uint16_t c = static_cast<uint16_t>(cell.col);
                  if (r >= tileRowMin && r <= tileRowMax && c >= tileColMin && c <= tileColMax) {
                    if (cell.digitIdx < digitLabels->getIndexedSize()) {
                      const auto& lbls = digitLabels->getLabels(cell.digitIdx);
                      for (const auto& lbl : lbls) {
                        clusterLabels->addElement(clsIdx, lbl);
                      }
                    }
                  }
                }
              }

              // Create O2 cluster for this tile
              o2::trk::Cluster cluster;
              cluster.chipID = chipID;
              cluster.row = tileRowMin;
              cluster.col = tileColMin;
              cluster.size = static_cast<uint16_t>(tileCells.size());
              if (geom) {
                cluster.subDetID = static_cast<int16_t>(geom->getSubDetID(chipID));
                cluster.layer = static_cast<int16_t>(geom->getLayer(chipID));
                cluster.disk = static_cast<int16_t>(geom->getDisk(chipID));
              }
              clusters.emplace_back(cluster);
            }
          }
        } else {
          // Normal cluster - encode directly
          std::array<unsigned char, o2::itsmft::ClusterPattern::MaxPatternBytes> patt{};
          for (const auto& cell : actsCluster.cells) {
            uint32_t ir = static_cast<uint32_t>(cell.row - rowMin);
            uint32_t ic = static_cast<uint32_t>(cell.col - colMin);
            int nbit = ir * colSpan + ic;
            patt[nbit >> 3] |= (0x1 << (7 - (nbit % 8)));
          }
          patterns.emplace_back(static_cast<unsigned char>(rowSpan));
          patterns.emplace_back(static_cast<unsigned char>(colSpan));
          const int nBytes = (rowSpan * colSpan + 7) / 8;
          patterns.insert(patterns.end(), patt.begin(), patt.begin() + nBytes);

          // Handle MC labels
          if (clusterLabels && digitLabels) {
            const auto clsIdx = static_cast<uint32_t>(clusters.size());
            for (const auto& cell : actsCluster.cells) {
              if (cell.digitIdx < digitLabels->getIndexedSize()) {
                const auto& lbls = digitLabels->getLabels(cell.digitIdx);
                for (const auto& lbl : lbls) {
                  clusterLabels->addElement(clsIdx, lbl);
                }
              }
            }
          }

          // Create O2 cluster
          o2::trk::Cluster cluster;
          cluster.chipID = chipID;
          cluster.row = rowMin;
          cluster.col = colMin;
          cluster.size = static_cast<uint16_t>(actsCluster.cells.size());
          if (geom) {
            cluster.subDetID = static_cast<int16_t>(geom->getSubDetID(chipID));
            cluster.layer = static_cast<int16_t>(geom->getLayer(chipID));
            cluster.disk = static_cast<int16_t>(geom->getDisk(chipID));
          }
          clusters.emplace_back(cluster);
        }
      }

      LOG(debug) << "    clusterization of chip " << chipID << " completed!";
    }
    clusterROFs.emplace_back(inROF.getBCData(), inROF.getROFrame(),
                             outFirst, static_cast<int>(clusters.size()) - outFirst);
  }

  if (clusterMC2ROFs && !digMC2ROFs.empty()) {
    clusterMC2ROFs->reserve(clusterMC2ROFs->size() + digMC2ROFs.size());
    for (const auto& in : digMC2ROFs) {
      clusterMC2ROFs->emplace_back(in.eventRecordID, in.rofRecordID, in.minROF, in.maxROF);
    }
  }
}
