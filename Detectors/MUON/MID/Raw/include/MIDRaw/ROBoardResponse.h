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

/// \file   MIDRaw/ROBoardResponse.h
/// \brief  Local board response
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 November 2021
#ifndef O2_MID_ROBOARDRESPONSE_H
#define O2_MID_ROBOARDRESPONSE_H

#include <vector>
#include "DataFormatsMID/ROBoard.h"
#include "MIDRaw/ROBoardConfigHandler.h"

namespace o2
{
namespace mid
{
class ROBoardResponse
{
 public:
  /// Default constructor
  ROBoardResponse() = default;
  /// Constructor from list of local board configuration
  ROBoardResponse(const std::vector<ROBoardConfig>& configurations);
  /// Default destructor
  ~ROBoardResponse() = default;

  /// Sets from configuration
  /// \param configurations Vector of local board configurations
  void set(const std::vector<ROBoardConfig>& configurations) { mConfigHandler.set(configurations); }

  /// Returns true if the local board has no fired digits after zero suppression
  bool isZeroSuppressed(const ROBoard& loc) const;

  /// Applies zero suppression
  /// \param locs Vector of local boards to be checked: zero suppressed boards are removed
  /// \return True if some board was removed
  bool applyZeroSuppression(std::vector<ROBoard>& locs) const;

  /// Returns the regional response
  /// \param locs Local board of one GBT link
  /// \return Vector with regional response
  std::vector<ROBoard> getRegionalResponse(const std::vector<ROBoard>& locs) const;

  /// Returns the trigger response
  /// \param triggerWord Trigger word
  /// \return List of trigger boards
  std::vector<ROBoard> getTriggerResponse(uint8_t triggerWord) const;

 private:
  ROBoardConfigHandler mConfigHandler; /// Local board configuration handler
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_ROBOARDRESPONSE_H */
