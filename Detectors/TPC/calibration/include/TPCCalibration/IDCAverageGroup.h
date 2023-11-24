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

/// \file IDCAverageGroup.h
/// \brief class for averaging and grouping of IDCs
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_IDCAVERAGEGROUP_H_
#define ALICEO2_IDCAVERAGEGROUP_H_

#include "TPCCalibration/RobustAverage.h"
#include "TPCCalibration/IDCAverageGroupBase.h"
#include "TPCBase/Sector.h"
#include "TPCBase/CalDet.h"
#include <boost/property_tree/ptree.hpp>
#include <vector>
#include "Rtypes.h"

namespace o2::utils
{
class TreeStreamRedirector;
}

namespace o2::tpc
{

// forward declaration of helper class
template <class Type>
class IDCAverageGroupHelper;

/// class for averaging and grouping IDCs
/// usage:
/// 1. Define grouping parameters
/// const int region = 3;
/// IDCAverageGroup<IDCAverageGroupCRU> idcaverage(6, 4, 3, 2, 111, region);
/// 2. set the ungrouped IDCs for one CRU
/// const int nIntegrationIntervals = 3;
/// std::vector<float> idcsungrouped(nIntegrationIntervals*Mapper::PADSPERREGION[region], 11.11); // vector containing IDCs for one region
/// idcaverage.setIDCs(idcsungrouped)
/// 3. perform the averaging and grouping
/// idcaverage.processIDCs();
/// 4. draw IDCs
/// idcaverage.drawUngroupedIDCs(0)
/// idcaverage.drawGroupedIDCs(0)
/// \tparam IDCAverageGroupCRU or IDCAverageGroupTPC
template <class Type>
class IDCAverageGroup : public IDCAverageGroupBase<Type>
{
 public:
  /// constructor
  /// \param groupPads number of pads in pad direction which will be grouped
  /// \param groupRows number of pads in row direction which will be grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are grouped differently. First digit specifies the EdgePadGroupingMethod  (example: 0: no pads are grouped, 110: first two pads are not grouped, 3210: first pad is not grouped, second + third pads are grouped, fourth + fifth + sixth pads are grouped)
  /// \param cru cru index
  /// \param overlapRows define parameter for additional overlapping pads in row direction
  /// \param overlapPads define parameter for additional overlapping pads in pad direction
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, IDCAverageGroupCRU>::value)), int>::type = 0>
  IDCAverageGroup(const unsigned char groupPads = 4, const unsigned char groupRows = 4, const unsigned char groupLastRowsThreshold = 2, const unsigned char groupLastPadsThreshold = 2, const unsigned int groupPadsSectorEdges = 0, const unsigned short cru = 0, const unsigned char overlapRows = 0, const unsigned char overlapPads = 0)
    : IDCAverageGroupBase<Type>{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges, cru, sNThreads}, mOverlapRows{overlapRows}, mOverlapPads{overlapPads}
  {
    init();
  }

  /// \param groupPads number of pads in pad direction which will be grouped
  /// \param groupRows number of pads in row direction which will be grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are grouped differently. First digit specifies the EdgePadGroupingMethod  (example: 0: no pads are grouped, 110: first two pads are not grouped, 3210: first pad is not grouped, second + third pads are grouped, fourth + fifth + sixth pads are grouped)
  /// \param overlapRows define parameter for additional overlapping pads in row direction
  /// \param overlapPads define parameter for additional overlapping pads in pad direction
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, IDCAverageGroupTPC>::value)), int>::type = 0>
  IDCAverageGroup(const std::array<unsigned char, Mapper::NREGIONS>& groupPads = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, const std::array<unsigned char, Mapper::NREGIONS>& groupRows = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold = {}, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold = {}, const unsigned int groupPadsSectorEdges = 0, const unsigned char overlapRows = 0, const unsigned char overlapPads = 0)
    : IDCAverageGroupBase<Type>{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupPadsSectorEdges, sNThreads}, mOverlapRows{overlapRows}, mOverlapPads{overlapPads}
  {
    init();
  }

  /// grouping and averaging of IDCs
  /// \param padStatusFlags pointer to map containing status flags for each pad to skip dead pads etc.
  void processIDCs(const CalDet<PadFlags>* padStatusFlags = nullptr);

  /// draw plot with information about the performed grouping
  /// \param filename name of the output file. If empty the name is chosen automatically
  void drawGrouping(const std::string filename = "");

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "IDCAverageGroup.root", const char* outName = "IDCAverageGroup") const;

  /// get the number of threads used for some of the calculations
  static int getNThreads() { return sNThreads; }

  /// \return returns grouped IDC object
  const auto& getIDCGroup() const { return this->mIDCsGrouped; }

  /// \return returns ungrouped IDCs
  const auto& getIDCsUngrouped() const { return this->mIDCsUngrouped; }

  /// \param sigma sigma which is used during outlier filtering
  static void setSigma(const float sigma) { o2::conf::ConfigurableParam::setValue<float>("TPCIDCGroupParam", "Sigma", sigma); }

  /// set the number of threads used for some of the calculations
  static void setNThreads(const int nThreads) { sNThreads = nThreads; }

  /// load ungrouped and grouped IDCs from File
  /// \param fileName name of the input file
  /// \param name name of the object in the output file
  bool setFromFile(const char* fileName = "IDCAverageGroup.root", const char* name = "IDCAverageGroup");

  /// for debugging: creating debug tree
  /// \param nameFile name of the output file
  void createDebugTree(const char* nameFile);

  /// for debugging: creating debug tree for integrated IDCs for all objects which are in the same file
  /// \param nameFile name of the output file
  /// \param filename name of the input file containing all objects
  static void createDebugTreeForAllCRUs(const char* nameFile, const char* filename);

 private:
  inline static int sNThreads{1};      ///< number of threads which are used during the calculations
  const unsigned char mOverlapRows{0}; ///< additional/overlapping pads in row direction (TODO overlap per region)
  const unsigned char mOverlapPads{0}; ///< additional/overlapping pads in pad direction (TODO overlap per region)

  /// init function
  void init();

  /// called from createDebugTreeForAllCRUs()
  static void createDebugTree(const IDCAverageGroupHelper<Type>& idcStruct, o2::utils::TreeStreamRedirector& pcstream);

  /// normal distribution used for weighting overlapping pads
  /// \param x distance to the center of the normal distribution
  /// \param sigma sigma of the normal distribution
  static float normal_dist(const float x, const float sigma);

  /// perform the loop over the IDCs by either perform the grouping or the drawing
  /// \param type containing necessary methods for either perform the grouping or the drawing
  template <class LoopType>
  void loopOverGroups(IDCAverageGroupHelper<LoopType>& idcStruct, const CalDet<PadFlags>* padStatusFlags = nullptr);

  /// draw information of the grouping on the pads (grouping parameters and number of grouped pads)
  void drawGroupingInformations(const int region, const int grPads, const int grRows, const int groupLastRowsThreshold, const int groupLastPadsThreshold, const int overlapRows, const int overlapPads, const int nIDCs, const int groupPadsSectorEdges) const;

  /// Helper function for drawing
  void drawLatex(IDCAverageGroupHelper<IDCAverageGroupDraw>& idcStruct, const GlobalPadNumber padNum, const unsigned int padInRegion, const bool fillPoly, const int colOffs = 0) const;

  ClassDefNV(IDCAverageGroup, 1)
};

} // namespace o2::tpc

#endif
