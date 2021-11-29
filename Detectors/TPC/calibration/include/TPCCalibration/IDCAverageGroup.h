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

template <typename T>
struct Enable_enum_class_bitfield {
  static constexpr bool value = false;
};

// operator overload for allowing bitfiedls with enum
template <typename T>
typename std::enable_if<std::is_enum<T>::value && Enable_enum_class_bitfield<T>::value, T>::type
  operator&(T lhs, T rhs)
{
  typedef typename std::underlying_type<T>::type integer_type;
  return static_cast<T>(static_cast<integer_type>(lhs) & static_cast<integer_type>(rhs));
}

template <typename T>
typename std::enable_if<std::is_enum<T>::value && Enable_enum_class_bitfield<T>::value, T>::type
  operator|(T lhs, T rhs)
{
  typedef typename std::underlying_type<T>::type integer_type;
  return static_cast<T>(static_cast<integer_type>(lhs) | static_cast<integer_type>(rhs));
}

enum class PadFlags : unsigned short {
  flagGoodPad = 1 << 0,     ///< flag for a good pad binary 0001
  flagDeadPad = 1 << 1,     ///< flag for a dead pad binary 0010
  flagUnknownPad = 1 << 2,  ///< flag for unknown status binary 0100
  flagSaturatedPad = 1 << 3 ///< flag for unknown status binary 0100
};

template <>
struct Enable_enum_class_bitfield<PadFlags> {
  static constexpr bool value = true;
};

/// class for averaging and grouping IDCs
/// usage:
/// 1. Define grouping parameters
/// const int region = 3;
/// IDCAverageGroup<IDCAverageGroupCRU> idcaverage(6, 4, 3, 2, region);
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
  /// \param region region of the TPC
  /// \param overlapRows define parameter for additional overlapping pads in row direction
  /// \param overlapPads define parameter for additional overlapping pads in pad direction
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, IDCAverageGroupCRU>::value)), int>::type = 0>
  IDCAverageGroup(const unsigned char groupPads = 4, const unsigned char groupRows = 4, const unsigned char groupLastRowsThreshold = 2, const unsigned char groupLastPadsThreshold = 2, const unsigned char groupNotnPadsSectorEdges = 0, const unsigned int region = 0, const Sector sector = Sector{0}, const unsigned char overlapRows = 0, const unsigned char overlapPads = 0)
    : IDCAverageGroupBase<Type>{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupNotnPadsSectorEdges, region, sector, sNThreads}, mOverlapRows{overlapRows}, mOverlapPads{overlapPads}
  {
    init();
  }

  /// \param groupPads number of pads in pad direction which will be grouped
  /// \param groupRows number of pads in row direction which will be grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  /// \param overlapRows define parameter for additional overlapping pads in row direction
  /// \param overlapPads define parameter for additional overlapping pads in pad direction
  template <bool IsEnabled = true, typename std::enable_if<(IsEnabled && (std::is_same<Type, IDCAverageGroupTPC>::value)), int>::type = 0>
  IDCAverageGroup(const std::array<unsigned char, Mapper::NREGIONS>& groupPads = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, const std::array<unsigned char, Mapper::NREGIONS>& groupRows = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold = {}, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold = {}, const unsigned char groupNotnPadsSectorEdges = 0, const unsigned char overlapRows = 0, const unsigned char overlapPads = 0)
    : IDCAverageGroupBase<Type>{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupNotnPadsSectorEdges, sNThreads}, mOverlapRows{overlapRows}, mOverlapPads{overlapPads}
  {
    init();
  }

  /// Update pad flag map from a local file
  /// \param file file containing the caldet map  with the flags
  // \param objName name of the object (TODO use a fixed name)
  void updatePadStatusMapFromFile(const char* file, const char* objName);

  /// TODO: Update pad flag map from the CCDB
  // void updatePadStatusMapFromCCDB(const char* file, const char* objName);

  /// grouping and averaging of IDCs
  void processIDCs();

  /// draw plot with information about the performed grouping
  /// \param filename name of the output file. If empty the name is chosen automatically
  void drawGrouping(const std::string filename = "");

  /// dump object to disc
  /// \param outFileName name of the output file
  /// \param outName name of the object in the output file
  void dumpToFile(const char* outFileName = "IDCAverageGroup.root", const char* outName = "IDCAverageGroup") const;

  /// draw the status map for the flags (for debugging) for a sector
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawPadStatusMapSector(const unsigned int sector, const std::string filename = "PadStatusFlags_Sector.pdf") const { drawPadStatusMap(false, Sector(sector), filename); }

  /// draw the status map for the flags (for debugging) for a full side
  /// \param side side which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawPadStatusMapSide(const o2::tpc::Side side, const std::string filename = "PadStatusFlags_Side.pdf") const { drawPadStatusMap(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename); }

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

  /// Set pad flag map directly
  /// \param padStatus CalDet containing for each pad the status flag
  void setPadStatusMap(const CalDet<PadFlags>& padStatus) { mPadStatus = std::make_unique<CalDet<PadFlags>>(padStatus); }

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
  inline static int sNThreads{1};                                                                                                 ///< number of threads which are used during the calculations
  const unsigned char mOverlapRows{0};                                                                                            ///< additional/overlapping pads in row direction (TODO overlap per region)
  const unsigned char mOverlapPads{0};                                                                                            ///< additional/overlapping pads in pad direction (TODO overlap per region)
  std::unique_ptr<CalDet<PadFlags>> mPadStatus{std::make_unique<CalDet<PadFlags>>(CalDet<PadFlags>("flags", PadSubset::Region))}; ///< status flag for each pad (i.e. if the pad is dead)

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
  void loopOverGroups(IDCAverageGroupHelper<LoopType>& idcStruct);

  /// helper function for drawing
  void drawPadStatusMap(const bool type, const Sector sector, const std::string filename) const;

  /// draw information of the grouping on the pads (grouping parameters and number of grouped pads)
  void drawGroupingInformations(const int region, const int grPads, const int grRows, const int groupLastRowsThreshold, const int groupLastPadsThreshold, const int overlapRows, const int overlapPads, const int nIDCs, const int groupPadsSectorEdges) const;

  ClassDefNV(IDCAverageGroup, 1)
};

} // namespace o2::tpc

#endif
