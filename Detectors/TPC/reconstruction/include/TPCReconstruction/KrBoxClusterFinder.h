// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file KrBoxClusterFinder3D.h
/// \brief Class for Krypton and X-ray events
/// \author Philip Hauer <hauer@hiskp.uni-bonn.de>

/// Based on old BoxClusterer.cxx which worked in two dimensions
///
/// The KrBoxClusterFinder3D basically tries to find local maximums and builds up
/// clusters around these. So if c is the local maximum (c for center). It
/// will try to make a cluster like this (shown here only in two dimensions):
///    --->  pad direction
///    o o o o o    |
///    o i i i o    |
///    o i C i o    V Time direction
///    o i i i o
///    o o o o o
///
/// The outer pad-time cells are only addded if the inner cell has signal. For
/// horizonal vertically aligned inner cells we test like this:
///        o
///        i
///    o i C i o
///        i
///        o
/// For the diagonal cells we check like this.
///    o o   o o
///    o i   i o
///        C
///    o i   i o
///    o o   o o
///
/// The requirements for a local maxima is:
/// Charge in bin is >= mQThresholdMax ADC channels.
/// Charge in bin is larger than all digits in the box around it.
/// (in the case when it is similar only one of the clusters will be made, see
/// in the code)
/// At least mMinNumberOfNeighbours neighbours (= direct neighbours) has a signal.
///
/// Implementation:
/// The RAW data is "expanded" for each sector and stored in a big signal
/// mMapOfAllDigits = 3d vector of all digits
///
/// To make sure that one never has to check if one is inside the sector or not
/// the arrays are larger than a sector. For the size in row-direction, a constant is used.
/// For time and pad direction, a number is set (here is room for improvements).
///
/// When data from a new sector is encountered, the method
///
/// How to use:
/// Load tpcdigits.root
/// Loop over all events
/// Loop over all sectors
/// Use KrBoxClusterFinder(sector) to create 3D map
/// Use localMaxima() to find..well the local maxima
/// localMaxima() yields a vector of a tuple with three int
/// Loop over this vector and give the tuple to buildCluster()
/// Write to tree or something similar

#ifndef ALICEO2_TPC_KrBoxClusterFinder_H_
#define ALICEO2_TPC_KrBoxClusterFinder_H_

#include "DataFormatsTPC/Digit.h"
#include "TPCReconstruction/KrCluster.h"

#include <tuple>
#include <vector>
#include <array>

namespace o2
{
namespace tpc
{

/// KrBoxClusterFinder class can be used to analyze X-ray and Krypton data
///
/// It creates a 3D map of all digits in one event. Afterwards the Function
/// findLocalMaxima() can be used to get the coordinates of all local maxima.
/// Around these maxima, the cluster is then built with the function
/// buildCluster. In a box around the center (defined by maxClusterSize....) it
/// looks for digits that can be assigned to the cluster.

class KrBoxClusterFinder
{
 public:
  /// Constructor:
  explicit KrBoxClusterFinder(std::vector<o2::tpc::Digit>& eventSector); ///< Creates a 3D Map

  /// After the map is created, we look for local maxima with this function:
  std::vector<std::tuple<int, int, int>> findLocalMaxima();

  /// After finding the local maxima, we can now build clusters around it.
  /// It works according to the explanation at the beginning of the header file.
  KrCluster buildCluster(int clusterCenterPad, int clusterCenterRow, int clusterCenterTime);

 private:
  // These variables can be varied
  int mMaxClusterSizeTime = 3; ///< The "radius" of a cluster in time direction
  int mMaxClusterSizePad = 3;  ///< radius in pad direction
  int mMaxClusterSizeRow = 2;  ///< radius in row direction

  // Todo: Differentiate between different ROCS:
  // int mMaxClusterSizeRowIROC = 3;  // radius in row direction
  // int mMaxClusterSizeRowOROC1 = 2; // radius in row direction
  // int mMaxClusterSizeRowOROC2 = 2; // radius in row direction
  // int mMaxClusterSizeRowOROC3 = 1; // radius in row direction

  float mQThresholdMax = 10.0;    ///< the Maximum charge in a cluster must exceed this value or it is discarded
  float mQThreshold = 1.0;        ///< every charge which is added to a cluster must exceed this value or it is discarded
  int mMinNumberOfNeighbours = 1; ///< amount of direct neighbours required for a cluster maximum

  /// Maximum Map Dimensions
  /// Here is room for improvements
  static constexpr size_t MaxPads = 138;  ///< Size of the map in pad-direction
  static constexpr size_t MaxRows = 152;  ///< Size of the map in row-direction
  static constexpr size_t MaxTimes = 550; ///< Size of the map in time-direction

  KrCluster mTempCluster; ///< Used to save the cluster data

  /// Here the map is defined where all digits are temporarily stored
  std::array<std::array<std::array<float, MaxPads>, MaxRows>, MaxTimes> mMapOfAllDigits{};

  /// To update the temporary cluster, i.e. all digits are added here
  void updateTempCluster(float tempCharge, int tempPad, int tempRow, int tempTime);
  /// After all digits are assigned to the cluster, the mean and sigmas are calculated here
  void updateTempClusterFinal();

  /// Returns sign of val (in a crazy way)
  int signnum(int val) { return (0 < val) - (val < 0); }

  ClassDefNV(KrBoxClusterFinder, 0);
};

} // namespace tpc
} // namespace o2

#endif
