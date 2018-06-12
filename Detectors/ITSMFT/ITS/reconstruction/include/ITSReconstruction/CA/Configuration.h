// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Configuration.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_CONFIGURATION_H_
#define TRACKINGITSU_INCLUDE_CONFIGURATION_H_

#include <array>
#include <climits>
#include <vector>

#include "json.h"

#include "ITSReconstruction/CA/Constants.h"

namespace o2
{
namespace ITS
{
namespace CA
{

template <typename Param>
class Configuration : public Param
{
 public:
  static Configuration<Param>& getInstance()
  {
    static Configuration<Param> instance;
    return instance;
  }

 private:
  Configuration() = default;
  Configuration(const Configuration<Param>&) = delete;
  const Configuration<Param>& operator=(const Configuration<Param>&) = delete;
};

struct TrackingParameters {
  int CellMinimumLevel(int iteration);
  int NumberOfIterations();

  /// General parameters
  int ClusterSharing = 0;
  std::vector<int> MinTrackLength = { 7 };
  /// Trackleting cuts
  std::vector<float> TrackletMaxDeltaPhi = { 0.3f };
  std::vector<std::array<float, Constants::ITS::TrackletsPerRoad>> TrackletMaxDeltaZ = { { 0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f } };
  /// Cell finding cuts
  std::vector<float> CellMaxDeltaTanLambda = { 0.025f };
  std::vector<std::array<float, Constants::ITS::CellsPerRoad>> CellMaxDCA = { { 0.05f, 0.04f, 0.05f, 0.2f, 0.4f } };
  std::vector<float> CellMaxDeltaPhi = { 0.14f };
  std::vector<std::array<float, Constants::ITS::CellsPerRoad>> CellMaxDeltaZ = { { 0.2f, 0.4f, 0.5f, 0.6f, 3.0f } };
  /// Neighbour finding cuts
  std::vector<std::array<float, Constants::ITS::CellsPerRoad - 1>> NeighbourMaxDeltaCurvature = { { 0.008f, 0.0025f, 0.003f, 0.0035f } };
  std::vector<std::array<float, Constants::ITS::CellsPerRoad - 1>> NeighbourMaxDeltaN = { { 0.002f, 0.0090f, 0.002f, 0.005f } };
};

struct MemoryParameters {
  /// Memory coefficients
  int MemoryOffset = 256;
  std::vector<std::array<float, Constants::ITS::CellsPerRoad>> CellsMemoryCoefficients = { { 2.3208e-08f, 2.104e-08f, 1.6432e-08f, 1.2412e-08f, 1.3543e-08f } };
  std::vector<std::array<float, Constants::ITS::TrackletsPerRoad>> TrackletsMemoryCoefficients = { { 0.0016353f, 0.0013627f, 0.000984f, 0.00078135f, 0.00057934f, 0.00052217f } };
};

struct IndexTableParameters {
  IndexTableParameters();
  void ComputeInverseBinSizes();
  int ZBins = 20;
  int PhiBins = 20;
  float InversePhiBinSize = 20 / Constants::Math::TwoPi;
  std::array<float, Constants::ITS::LayersNumber> InverseZBinSize;
};

inline int TrackingParameters::NumberOfIterations()
{
  return MinTrackLength.size();
}

inline int TrackingParameters::CellMinimumLevel(int iteration)
{
  return MinTrackLength[iteration] - Constants::ITS::ClustersPerCell + 1;
}

inline IndexTableParameters::IndexTableParameters()
{
  ComputeInverseBinSizes();
}

inline void IndexTableParameters::ComputeInverseBinSizes()
{
  InversePhiBinSize = PhiBins / Constants::Math::TwoPi;
  for (int iL = 0; iL < Constants::ITS::LayersNumber; ++iL) {
    InverseZBinSize[iL] = 0.5f * ZBins / Constants::ITS::LayersZCoordinate()[iL];
  }
}

} // namespace CA
} // namespace ITS
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_CONFIGURATION_H_ */
