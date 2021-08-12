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
/// Old ansatz: Store every digit in a big 3D map
/// Problem: Memory consumption too large if timebins > 10k
/// New solution: Use rolling map aka "The set of timeslices"
///
///      1 2 3 4 5 6 7
///
///   1  o o o x o o o
///   2  o o o x o o o
///      .............
/// n-1  o o o x o o o
///   n  o o o x o o o
///
/// x-axis: timeslices
/// y-axis: global pad number
/// In this example, the fourth timeslice is the interesting one. Here, local maxima and clusters are found. After this slice has been processed, the first slice will be dropped and another slice will be added at the last position (seventh position).
/// Afterwards, the algorithm looks for clusters in the middle time slice.
///
/// How to use (see macro: findKrBoxCluster.C):
/// Create KrBoxClusterFinder object
/// Load gainmap (if available)
/// Make adjustments to clusterfinder (e.g. set min number of neighbours)
/// Loop over all sectors
/// Use "loopOverSector" function for this
/// Write to tree or something similar

#ifndef ALICEO2_TPC_KrBoxClusterFinder_H_
#define ALICEO2_TPC_KrBoxClusterFinder_H_

#include "DataFormatsTPC/Digit.h"
#include "DataFormatsTPC/KrCluster.h"
#include "TPCReconstruction/KrBoxClusterFinderParam.h"

#include "TPCBase/Mapper.h"

#include "TPCBase/CalDet.h"

#include <tuple>
#include <vector>
#include <array>
#include <deque>
#include <gsl/span>

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
 private:
  /// Maximum Map Dimensions
  /// Here is room for improvements
  static constexpr size_t MaxPads = 138; ///< Size of the map in pad-direction
  static constexpr size_t MaxRows = 152; ///< Size of the map in row-direction
  size_t mMaxTimes = 114048;             ///< Size of the map in time-direction

 public:
  /// Constructor:
  /// The constructor allocates a three dimensional array (Pad,Row,Time) which is
  /// later filled with the recorded charges for each digit
  explicit KrBoxClusterFinder() = default;

  explicit KrBoxClusterFinder(int sector) : mSector(sector){};

  /// If a gain map exists, the map can be loaded with this function
  /// The function expects a CalDet file with a gain map (gain entry for each pad).
  void loadGainMapFromFile(const std::string_view calDetFileName, const std::string_view gainMapName = "GainMap");

  /// After the map is created, we look for local maxima with this function:
  std::vector<std::tuple<int, int, int>> findLocalMaxima(bool directFilling = true, const int timeOffset = 0);

  /// After finding the local maxima, we can now build clusters around it.
  /// It works according to the explanation at the beginning of the header file.
  KrCluster buildCluster(int clusterCenterPad, int clusterCenterRow, int clusterCenterTime, bool directFilling = false, const int timeOffset = 0);

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

  /// Set Function for minimal charge required for maxCharge of a cluster
  void setMaxTimes(int maxTimes) { mMaxTimes = maxTimes; }

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

  void loopOverSector(const gsl::span<const Digit> eventSector, const int sector);

  void loopOverSector(const std::vector<Digit>& eventSector, const int sector)
  {
    loopOverSector(gsl::span(eventSector.data(), eventSector.size()), sector);
  }

 private:
  // These variables can be varied
  // They were choses such that the box in each readout chamber is approx. the same size
  // NOTE: They will be overwritten by the values in KrBoxClusterFinderParam in case the init() function is called
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

  float mCutMinSigmaTime{0};      ///< Min sigma time to accept cluster
  float mCutMaxSigmaTime{1000};   ///< Min sigma time to accept cluster
  float mCutMinSigmaPad{0};       ///< Min sigma pad to accept cluster
  float mCutMaxSigmaPad{1000};    ///< Min sigma pad to accept cluster
  float mCutMinSigmaRow{0};       ///< Min sigma row to accept cluster
  float mCutMaxSigmaRow{1000};    ///< Min sigma row to accept cluster
  float mCutMaxQtot{1e10};        ///< Max Qtot to accept cluster
  float mCutQtot0{1e10};          ///< Max Qtot at zero size for Qtot vs. size correlation cut
  float mCutQtotSizeSlope{0};     ///< Max Qtot over size slope for Qtot vs. size correlation cut
  unsigned char mCutMaxSize{255}; ///< Max cluster size in number of digits
  bool mApplyCuts{false};         ///< if to apply cluster cuts above

  int mSector = -1;                 ///< sector being processed in this instance
  std::unique_ptr<CalPad> mGainMap; ///< Gain map object

  /// Values to define ROC boundaries
  static constexpr size_t MaxRowsIROC = 63;  ///< Amount of rows in IROC
  static constexpr size_t MaxRowsOROC1 = 34; ///< Amount of rows in OROC1
  static constexpr size_t MaxRowsOROC2 = 30; ///< Amount of rows in OROC2
  static constexpr size_t MaxRowsOROC3 = 25; ///< Amount of rows in OROC3

  /// Vector of cluster objects
  /// Used for returning results
  std::vector<o2::tpc::KrCluster> mClusters;

  /// Need an instance of Mapper to know position of pads
  const Mapper& mMapperInstance = o2::tpc::Mapper::instance();

  KrCluster mTempCluster; ///< Used to save the cluster data

  using TimeSliceSector = std::array<std::array<float, MaxPads>, MaxRows>;

  /// A temporary timeslice that gets filled and can be added to the set of timeslices
  // TimeSliceSector mTempTimeSlice;

  /// Used to keep track of the digit that has to be processed
  size_t mFirstDigit = 0;

  /// The set of timeslices.
  /// It consists of 2*mMaxClusterSizeTime + 1 timeslices.
  /// You can imagine it like this:
  /// \verbatim
  ///
  ///      1 2 3 4 5 6 7
  ///
  ///   1  o o o x o o o
  ///   2  o o o x o o o
  ///      .............
  /// n-1  o o o x o o o
  ///   n  o o o x o o o
  ///
  /// \endverbatim
  ///
  /// x-axis: Timeslice number
  /// y-axis: Pad number
  /// Time slice four is the interesting one. In there, local maxima are found and clusters are built from it. After it is processed, timeslice number 1 will be dropped and another timeslice will be put at the end of the set.
  std::deque<TimeSliceSector> mSetOfTimeSlices{};

  /// pre-store if complete time bins and rows within time bins have charges above mQThresholdMax
  struct ThresholdInfo {
    bool digitAboveThreshold{};
    std::array<bool, MaxRows> rowAboveThreshold{};
  };

  std::deque<ThresholdInfo> mThresholdInfo{};

  void createInitialMap(const gsl::span<const Digit> eventSector);
  void popFirstTimeSliceFromMap()
  {
    mSetOfTimeSlices.pop_front();
    mThresholdInfo.pop_front();
  }
  void fillADCValueInLastSlice(int cru, int rowInSector, int padInRow, float adcValue);
  void addTimeSlice(const gsl::span<const Digit> eventSector, const int timeSlice);

  /// For each ROC, the maximum cluster size has to be chosen
  void setMaxClusterSize(int row);

  /// To update the temporary cluster, i.e. all digits are added here
  void updateTempCluster(float tempCharge, int tempPad, int tempRow, int tempTime);
  /// After all digits are assigned to the cluster, the mean and sigmas are calculated here
  void updateTempClusterFinal(const int timeOffset = 0);

  /// Returns sign of val (in a crazy way)
  int signnum(int val) { return (0 < val) - (val < 0); }

  /// Cluster acceptance cuts
  bool acceptCluster(const KrCluster& cl);

  ClassDefNV(KrBoxClusterFinder, 0);
};

} // namespace tpc
} // namespace o2

#endif
