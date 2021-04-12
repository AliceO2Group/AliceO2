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
/// ToDo: Find an elegant way to split the huge map into four (IROC, OROC1, OROC2 and OROC3) smaller maps. Unfortunately, this seems to interfere with the rest of the code.
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
#include "TPCReconstruction/KrBoxClusterFinderParam.h"

#include "TPCBase/Mapper.h"

#include "TPCBase/CalDet.h"

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
  /// The constructor allocates a three dimensional array (Pad,Row,Time) which is
  /// later filled with the recorded charges for each digit
  explicit KrBoxClusterFinder() = default;

  explicit KrBoxClusterFinder(int sector) : mSector(sector){};

  /// If a gain map exists, the map can be loaded with this function
  /// The function expects a CalDet file with a gain map (gain entry for each pad).
  void loadGainMapFromFile(const std::string_view calDetFileName, const std::string_view gainMapName = "GainMap");

  /// Function used in macro to fill the map with all recorded digits
  void fillAndCorrectMap(std::vector<o2::tpc::Digit>& eventSector, const int sector);

  /// Function to fill single digi
  void fillADCValue(int cru, int rowInSector, int padInRow, int timeBin, float adcValue);

  /// After the map is created, we look for local maxima with this function:
  std::vector<std::tuple<int, int, int>> findLocalMaxima(bool directFilling = false);

  /// After finding the local maxima, we can now build clusters around it.
  /// It works according to the explanation at the beginning of the header file.
  KrCluster buildCluster(int clusterCenterPad, int clusterCenterRow, int clusterCenterTime, bool directFilling = false);

  /// reset the ADC map
  void resetADCMap();

  /// reset cluster vector
  void resetClusters() { mClusters.clear(); }

  std::vector<KrCluster>& getClusters() { return mClusters; }
  const std::vector<KrCluster>& getClusters() const { return mClusters; }

  /// set sector of this instance
  void setSector(int sector) { mSector = sector; }

  /// ger sector of this instance
  int getSector() const { return mSector; }

  /// initialize the parameters from KrBoxClusterFinderParam
  void init();

  /// Set Function for minimum number of direct neighbours required
  void setMinNumberOfNeighbours(int minNumberOfNeighbours) { mMinNumberOfNeighbours = minNumberOfNeighbours; }

  /// Set Function for minimal charge required for maxCharge of a cluster
  void setMinQTreshold(int minQThreshold) { mQThresholdMax = minQThreshold; }

  /// Set Function for maximal cluster sizes in different ROCs
  void setMaxClusterSize(int maxClusterSizeRowIROC, int maxClusterSizeRowOROC1, int maxClusterSizeRowOROC2, int maxClusterSizeRowOROC3,
                         int maxClusterSizePadIROC, int maxClusterSizePadOROC1, int maxClusterSizePadOROC2, int maxClusterSizePadOROC3,
                         int maxClusterSizeTime)
  {
    mMaxClusterSizeRowIROC = maxClusterSizeRowIROC;
    mMaxClusterSizeRowOROC1 = maxClusterSizeRowOROC1;
    mMaxClusterSizeRowOROC2 = maxClusterSizeRowOROC2;
    mMaxClusterSizeRowOROC3 = maxClusterSizeRowOROC3;

    mMaxClusterSizePadIROC = maxClusterSizePadIROC;
    mMaxClusterSizePadOROC1 = maxClusterSizePadOROC1;
    mMaxClusterSizePadOROC2 = maxClusterSizePadOROC2;
    mMaxClusterSizePadOROC3 = maxClusterSizePadOROC3;

    mMaxClusterSizeTime = maxClusterSizeTime;
  }

 private:
  // These variables can be varied
  // They were choses such that the box in each readout chamber is approx. the same size
  int mMaxClusterSizeTime = 3; ///< The "radius" of a cluster in time direction
  int mMaxClusterSizeRow;      ///< The "radius" of a cluster in row direction
  int mMaxClusterSizePad;      ///< The "radius" of a cluster in pad direction

  int mMaxClusterSizeRowIROC = 3;  ///< The "radius" of a cluster in row direction in IROC
  int mMaxClusterSizeRowOROC1 = 2; ///< The "radius" of a cluster in row direction in OROC1
  int mMaxClusterSizeRowOROC2 = 2; ///< The "radius" of a cluster in row direction in OROC2
  int mMaxClusterSizeRowOROC3 = 1; ///< The "radius" of a cluster in row direction in OROC3

  int mMaxClusterSizePadIROC = 5;  ///< The "radius" of a cluster in pad direction in IROC
  int mMaxClusterSizePadOROC1 = 3; ///< The "radius" of a cluster in pad direction in OROC1
  int mMaxClusterSizePadOROC2 = 3; ///< The "radius" of a cluster in pad direction in OROC2
  int mMaxClusterSizePadOROC3 = 3; ///< The "radius" of a cluster in pad direction in OROC3

  float mQThresholdMax = 30.0;    ///< the Maximum charge in a cluster must exceed this value or it is discarded
  float mQThreshold = 1.0;        ///< every charge which is added to a cluster must exceed this value or it is discarded
  int mMinNumberOfNeighbours = 2; ///< amount of direct neighbours required for a cluster maximum

  int mSector = -1;                 ///< sector being processed in this instance
  std::unique_ptr<CalPad> mGainMap; ///< Gain map object

  /// Maximum Map Dimensions
  /// Here is room for improvements
  static constexpr size_t MaxPads = 138;    ///< Size of the map in pad-direction
  static constexpr size_t MaxRows = 152;    ///< Size of the map in row-direction
  static constexpr size_t MaxTimes = 20000; ///< Size of the map in time-direction

  /// Values to define ROC boundaries
  static constexpr size_t MaxRowsIROC = 63;  ///< Amount of rows in IROC
  static constexpr size_t MaxRowsOROC1 = 34; ///< Amount of rows in OROC1
  static constexpr size_t MaxRowsOROC2 = 30; ///< Amount of rows in OROC2
  static constexpr size_t MaxRowsOROC3 = 25; ///< Amount of rows in OROC3

  std::vector<o2::tpc::KrCluster> mClusters;

  /// Need an instance of Mapper to know position of pads
  const Mapper& mMapperInstance = o2::tpc::Mapper::instance();

  KrCluster mTempCluster; ///< Used to save the cluster data

  /// Here the map is defined where all digits are temporarily stored
  std::array<std::array<std::array<float, MaxPads>, MaxRows>, MaxTimes> mMapOfAllDigits{};

  /// For each ROC, the maximum cluster size has to be chosen
  void setMaxClusterSize(int row);

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
