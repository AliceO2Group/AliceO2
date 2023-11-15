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

/// \file IDCGroupingParameter.h
/// \brief Definition of the parameter for the grouping of the IDCs
/// \author Matthias Kleiner, mkleiner@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_IDCGROUPINGPARAMETER_H_
#define ALICEO2_TPC_IDCGROUPINGPARAMETER_H_

#include <array>
#include "CommonUtils/ConfigurableParamHelper.h"
#include "TPCBase/Mapper.h"

namespace o2
{
namespace tpc
{

/// averaging methods which is used for averaging IDCs from grouped pads
enum class AveragingMethod : char {
  FAST = 0, ///< no outlier filtering performed. Only averaging
  SLOW = 1  ///< Outlier filtering performed. Filtering + averaging
};

enum class EdgePadGroupingMethod : char {
  NO = 0,  ///< no grouping in row direction
  ROWS = 1 ///< grouping in row direction
};

/// struct for setting the parameters for the grouping of IDCs
struct ParameterIDCGroup : public o2::conf::ConfigurableParamHelper<ParameterIDCGroup> {
  unsigned char groupPads[Mapper::NREGIONS]{7, 7, 7, 7, 6, 6, 6, 6, 5, 5};              ///< grouping parameter in pad direction (how many pads are grouped)
  unsigned char groupRows[Mapper::NREGIONS]{5, 5, 5, 5, 4, 4, 4, 4, 3, 3};              ///< group parameter in row direction (how many rows are grouped)
  unsigned char groupLastRowsThreshold[Mapper::NREGIONS]{3, 3, 3, 3, 2, 2, 2, 2, 2, 2}; ///< if the last group (region edges) consists in row direction less then mGroupLastRowsThreshold pads then it will be grouped into the previous group
  unsigned char groupLastPadsThreshold[Mapper::NREGIONS]{3, 3, 3, 3, 2, 2, 2, 2, 1, 1}; ///< if the last group (sector edges) consists in pad direction less then mGroupLastPadsThreshold pads then it will be grouped into the previous group
  unsigned int groupPadsSectorEdges{0};                                                 ///< decoded number of pads at the sector edges which are grouped differently. First digit specifies the EdgePadGroupingMethod  (example: 0: no pads are grouped, 110: first two pads are not grouped, 3210: first pad is not grouped, second + third pads are grouped, fourth + fifth + sixth pads are grouped)
  AveragingMethod method = AveragingMethod::FAST;                                       ///< method which is used for averaging
  float sigma = 3.f;                                                                    ///< sigma cut which can be used during the grouping for outlier filtering
  float minIDC0Median = 8;                                                              ///< this value is used for identifying outliers (pads with low IDC0 values): "accepted IDC 0 values > median_IDC0 + stdDev * minIDC0Median"
  float maxIDC0Median = 15;                                                             ///< this value is used for identifying outliers (pads with high IDC0 values): "accepted IDC 0 values < median_IDC0 + stdDev * maxIDC0Median"
  float minmaxIDC0MedianCentreEdge = 20;                                                ///< this value is used for identifying outliers at the centre of the cross and the edges of the sector i.e. where the gain is significantly lower (cross) and sometimes higher at the edges"

  /// Helper function for setting the groupimg parameters from a string (can be "X": parameters in all regions are "X" or can be "1,2,3,4,5,6,7,8,9,10" for setting individual regions)
  /// \param sgroupPads string for grouping parameter in pad direction
  /// \param sgroupRows string for grouping parameter in row direction
  /// \param sgroupLastRowsThreshold string for grouping parameter of last pads int row direction
  /// \param sgroupLastPadsThreshold string for grouping parameter of last pads int pad direction
  static void setGroupingParameterFromString(const std::string sgroupPads, const std::string sgroupRows, const std::string sgroupLastRowsThreshold, const std::string sgroupLastPadsThreshold);

  /// returns total number of pads which are grouped separately: calculate sum of all digits in integer value (example: 0: returns 0, 11: returns 2, 321: returns 6)
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are grouped differently
  static unsigned int getTotalGroupPadsSectorEdges(unsigned int groupPadsSectorEdges) { return getTotalGroupPadsSectorEdgesHelper(groupPadsSectorEdges / 10); }

  /// \return returns the number of differently grouped pads per row: returns sum of digits in integer whoch are not 0 (example: 0: returns 0, 11: returns 2, 321: returns 3)
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are grouped differently
  static unsigned int getGroupedPadsSectorEdges(unsigned int groupPadsSectorEdges) { return getGroupedPadsSectorEdgesHelper(groupPadsSectorEdges / 10); }

  /// \return returns how the edge pads are getting grouped
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are grouped differently
  static EdgePadGroupingMethod getEdgePadGroupingType(unsigned int groupPadsSectorEdges);

  /// \return returns number of ungrouped pads for grouped pad (example groupPadsSectorEdges=324530: group=0 -> 3, group=1 -> 5, group=2 -> 4...
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are grouped differently
  /// \param group index of the group
  static unsigned int getPadsInGroupSectorEdges(unsigned int groupPadsSectorEdges, const unsigned int group) { return getPadsInGroupSectorEdgesHelper(groupPadsSectorEdges / 10, group); }

 private:
  /// returns total number of pads which are grouped separately: calculate sum of all digits in integer value (example: 0: returns 0, 11: returns 2, 321: returns 6)
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are grouped differently
  static unsigned int getTotalGroupPadsSectorEdgesHelper(unsigned int groupPadsSectorEdges);

  /// \return returns the number of differently grouped pads per row: returns sum of digits in integer whoch are not 0 (example: 0: returns 0, 11: returns 2, 321: returns 3)
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are grouped differently
  static unsigned int getGroupedPadsSectorEdgesHelper(unsigned int groupPadsSectorEdges);

  /// \return returns number of ungrouped pads for grouped pad (example groupPadsSectorEdges=324530: group=0 -> 3, group=1 -> 5, group=2 -> 4...
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are grouped differently
  /// \param group index of the group
  static unsigned int getPadsInGroupSectorEdgesHelper(unsigned int groupPadsSectorEdges, const unsigned int group);

  O2ParamDef(ParameterIDCGroup, "TPCIDCGroupParam");
};

/// struct for storing the parameters for the grouping of IDCs to CCDB
struct ParameterIDCGroupCCDB {

  /// contructor
  /// \param groupPads number of pads in pad direction which are grouped
  /// \param groupRows number of pads in row direction which are grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  /// \param groupNotnPadsSectorEdges number of pads at the sector edges which are not getting grouped
  ParameterIDCGroupCCDB(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold, const unsigned int groupNotnPadsSectorEdges)
    : groupPads{groupPads}, groupRows{groupRows}, groupLastRowsThreshold{groupLastRowsThreshold}, groupLastPadsThreshold{groupLastPadsThreshold}, groupPadsSectorEdges{groupNotnPadsSectorEdges} {};

  ParameterIDCGroupCCDB() = default;

  /// \return returns number of pads in pad direction which are grouped
  /// \parameter region TPC region
  unsigned char getGroupPads(const unsigned int region) const { return groupPads[region]; }

  /// \return returns number of pads in row direction which are grouped
  /// \parameter region TPC region
  unsigned char getGroupRows(const unsigned int region) const { return groupRows[region]; }

  /// \return returns minimum number of pads in row direction for the last group in row direction
  /// \parameter region TPC region
  unsigned char getGroupLastRowsThreshold(const unsigned int region) const { return groupLastRowsThreshold[region]; }

  /// \return returns minimum number of pads in pad direction for the last group in pad direction
  /// \parameter region TPC region
  unsigned char getGroupLastPadsThreshold(const unsigned int region) const { return groupLastPadsThreshold[region]; }

  /// \return returns the number of pads at the sector edges which are not grouped
  unsigned int getGroupPadsSectorEdges() const { return groupPadsSectorEdges; }

  /// returns total number of pads which are grouped separately: calculate sum of all digits in integer value (example: 0: returns 0, 11: returns 2, 321: returns 6)
  unsigned int getTotalGroupPadsSectorEdges() const { return ParameterIDCGroup::getTotalGroupPadsSectorEdges(groupPadsSectorEdges); }

  /// \return returns the number of differently grouped pads per row: returns sum of digits in integer whoch are not 0 (example: 0: returns 0, 11: returns 2, 321: returns 3)
  /// \param groupPadsSectorEdges decoded number of pads at the sector edges which are grouped differently
  unsigned int getGroupedPadsSectorEdges() const { return ParameterIDCGroup::getGroupedPadsSectorEdges(groupPadsSectorEdges); }

  /// \return returns number of pads in pad direction which are grouped for all regions
  const std::array<unsigned char, Mapper::NREGIONS>& getGroupPads() const { return groupPads; }

  /// \return returns number of pads in row direction which are grouped for all regions
  const std::array<unsigned char, Mapper::NREGIONS>& getGroupRows() const { return groupRows; }

  /// \return returns minimum number of pads in row direction for the last group in row direction for all regions
  const std::array<unsigned char, Mapper::NREGIONS>& getGroupLastRowsThreshold() const { return groupLastRowsThreshold; }

  /// \return returns minimum number of pads in pad direction for the last group in pad direction for all regions
  const std::array<unsigned char, Mapper::NREGIONS>& getGroupLastPadsThreshold() const { return groupLastPadsThreshold; }

  /// \return returns which type of grouping is performed for the sector edge pads
  EdgePadGroupingMethod getEdgePadGroupingType() const { return ParameterIDCGroup::getEdgePadGroupingType(groupPadsSectorEdges); }

  /// \return returns number of ungrouped pads for grouped pad (example groupPadsSectorEdges=324530: group=0 -> 3, group=1 -> 5, group=2 -> 4...
  /// \param group index of the group
  unsigned int getPadsInGroupSectorEdges(const unsigned indexGroup) const { return ParameterIDCGroup::getPadsInGroupSectorEdges(groupPadsSectorEdges, indexGroup); }

  std::array<unsigned char, Mapper::NREGIONS> groupPads{};              ///< grouping parameter in pad direction (how many pads in pad direction are grouped)
  std::array<unsigned char, Mapper::NREGIONS> groupRows{};              ///< grouping parameter in pad direction (how many pads in pad direction are grouped)
  std::array<unsigned char, Mapper::NREGIONS> groupLastRowsThreshold{}; ///< if the last group (region edges) consists in row direction less then mGroupLastRowsThreshold pads then it will be grouped into the previous group
  std::array<unsigned char, Mapper::NREGIONS> groupLastPadsThreshold{}; ///< if the last group (sector edges) consists in pad direction less then mGroupLastPadsThreshold pads then it will be grouped into the previous group
  unsigned int groupPadsSectorEdges{0};                                 ///< decoded number of pads at the sector edges which are grouped differently (example: 0: no pads are grouped, 11: first two pads are not grouped, 321: first pad is not grouped, second + third pads are grouped, fourth + fifth + sixth pads are grouped)
};

struct ParameterIDCCompression : public o2::conf::ConfigurableParamHelper<ParameterIDCCompression> {
  float maxIDCDeltaValue = 3.f;  ///< maximum Delta IDC
  float minIDCDeltaValue = -1.f; ///< minimum Delta IDC
  O2ParamDef(ParameterIDCCompression, "TPCIDCCompressionParam");
};

} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_ParameterGEM_H_
