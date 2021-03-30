// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

/// struct for setting the parameters for the grouping of IDCs
struct ParameterIDCGroup : public o2::conf::ConfigurableParamHelper<ParameterIDCGroup> {
  unsigned char GroupPads[Mapper::NREGIONS]{7, 7, 7, 7, 6, 6, 6, 6, 5, 5};              ///< grouping parameter in pad direction (how many pads are grouped)
  unsigned char GroupRows[Mapper::NREGIONS]{5, 5, 5, 5, 4, 4, 4, 4, 3, 3};              ///< group parameter in row direction (how many rows are grouped)
  unsigned char GroupLastRowsThreshold[Mapper::NREGIONS]{3, 3, 3, 3, 2, 2, 2, 2, 2, 2}; ///< if the last group (region edges) consists in row direction less then mGroupLastRowsThreshold pads then it will be grouped into the previous group
  unsigned char GroupLastPadsThreshold[Mapper::NREGIONS]{3, 3, 3, 3, 2, 2, 2, 2, 1, 1}; ///< if the last group (sector edges) consists in pad direction less then mGroupLastPadsThreshold pads then it will be grouped into the previous group
  O2ParamDef(ParameterIDCGroup, "TPCIDCGroupParam");
};

/// struct for storing the parameters for the grouping of IDCs to CCDB
struct ParameterIDCGroupCCDB {

  /// contructor
  /// \param groupPads number of pads in pad direction which are grouped
  /// \param groupRows number of pads in row direction which are grouped
  /// \param groupLastRowsThreshold minimum number of pads in row direction for the last group in row direction
  /// \param groupLastPadsThreshold minimum number of pads in pad direction for the last group in pad direction
  ParameterIDCGroupCCDB(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold)
    : GroupPads{groupPads}, GroupRows{groupRows}, GroupLastRowsThreshold{groupLastRowsThreshold}, GroupLastPadsThreshold{groupLastPadsThreshold} {};

  ParameterIDCGroupCCDB() = default;

  /// \return returns number of pads in pad direction which are grouped
  /// \parameter region TPC region
  unsigned char getGroupPads(const unsigned int region) const { return GroupPads[region]; }

  /// \return returns number of pads in row direction which are grouped
  /// \parameter region TPC region
  unsigned char getGroupRows(const unsigned int region) const { return GroupRows[region]; }

  /// \return returns minimum number of pads in row direction for the last group in row direction
  /// \parameter region TPC region
  unsigned char getGroupLastRowsThreshold(const unsigned int region) const { return GroupLastRowsThreshold[region]; }

  /// \return returns minimum number of pads in pad direction for the last group in pad direction
  /// \parameter region TPC region
  unsigned char getGroupLastPadsThreshold(const unsigned int region) const { return GroupLastPadsThreshold[region]; }

  /// \return returns number of pads in pad direction which are grouped for all regions
  const std::array<unsigned char, Mapper::NREGIONS>& getGroupPads() const { return GroupPads; }

  /// \return returns number of pads in row direction which are grouped for all regions
  const std::array<unsigned char, Mapper::NREGIONS>& getGroupRows() const { return GroupRows; }

  /// \return returns minimum number of pads in row direction for the last group in row direction for all regions
  const std::array<unsigned char, Mapper::NREGIONS>& getGroupLastRowsThreshold() const { return GroupLastRowsThreshold; }

  /// \return returns minimum number of pads in pad direction for the last group in pad direction for all regions
  const std::array<unsigned char, Mapper::NREGIONS>& getGroupLastPadsThreshold() const { return GroupLastPadsThreshold; }

  std::array<unsigned char, Mapper::NREGIONS> GroupPads{};              ///< grouping parameter in pad direction (how many pads in pad direction are grouped)
  std::array<unsigned char, Mapper::NREGIONS> GroupRows{};              ///< grouping parameter in pad direction (how many pads in pad direction are grouped)
  std::array<unsigned char, Mapper::NREGIONS> GroupLastRowsThreshold{}; ///< if the last group (region edges) consists in row direction less then mGroupLastRowsThreshold pads then it will be grouped into the previous group
  std::array<unsigned char, Mapper::NREGIONS> GroupLastPadsThreshold{}; ///< if the last group (sector edges) consists in pad direction less then mGroupLastPadsThreshold pads then it will be grouped into the previous group
};

struct ParameterIDCCompression : public o2::conf::ConfigurableParamHelper<ParameterIDCCompression> {
  float MaxIDCDeltaValue = 0.3f; ///< maximum Delta IDC
  O2ParamDef(ParameterIDCCompression, "TPCIDCCompressionParam");
};

} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_ParameterGEM_H_
