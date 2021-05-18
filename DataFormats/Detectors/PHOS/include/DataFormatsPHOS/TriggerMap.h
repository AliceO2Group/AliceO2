// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class TRIGGERMAP
/// \brief CCDB container for trigger bad map and turn-on curves
/// \author Dmitri Peresunko, RRC Kurchatov institute
/// \since March 20, 2021
///
///

#ifndef PHOS_TRIGGERMAP_H
#define PHOS_TRIGGERMAP_H

#include <array>
#include <bitset>
#include "TObject.h"

class TH1;

namespace o2
{

namespace phos
{

class TriggerMap
{
 public:
  /// \brief Constructor
  TriggerMap() = default;

  /// \brief Constructor for tests
  TriggerMap(int test);

  TriggerMap& operator=(const TriggerMap& other) = default;

  /// \brief Destructor
  ~TriggerMap() = default;

  /// \brief tests if cell is in active trigger region
  /// \param cellID Absolute ID of cell
  /// \return true if cell is in active trigger region
  bool isGood2x2(short cellID) const { return !mTrigger2x2Map.test(cellID); }

  /// \brief Add bad triger cell to the container
  /// \param cellID Absolute ID of the bad channel
  void addBad2x2Channel(short cellID) { mTrigger2x2Map.set(cellID); } //set bit to true

  /// \brief Mark trigger channel as good
  /// \param cellID Absolute ID of the channel
  void set2x2ChannelGood(short cellID) { mTrigger2x2Map.set(cellID, false); }

  /// \brief tests if cell is in active trigger region
  /// \param cellID Absolute ID of cell
  /// \return true if cell is in active trigger region
  bool isGood4x4(short cellID) const { return !mTrigger4x4Map.test(cellID); }

  /// \brief Add bad triger cell to the container
  /// \param cellID Absolute ID of the bad channel
  void addBad4x4Channel(short cellID) { mTrigger4x4Map.set(cellID); } //set bit to true

  /// \brief Mark trigger channel as good
  /// \param cellID Absolute ID of the channel
  void set4x4ChannelGood(short cellID) { mTrigger2x2Map.set(cellID, false); }

  void setTurnOnCurvesVestion(int v = 0);

  /// \brief random return true with probability to fire trigger
  /// \param  a amplitude of trigger tile
  /// \param iTRU,ix,iz coordinates of trigger tile
  bool isFiredMC2x2(float a, short iTRU, short ix, short iz) const;

  /// \brief random return true with probability to fire trigger
  /// \param  a amplitude of trigger tile
  /// \param iTRU,ix,iz coordinates of trigger tile
  bool isFiredMC4x4(float a, short iTRU, short ix, short iz) const;

  bool try2x2(float a, short iTRU) const;
  bool try4x4(float a, short iTRU) const;

  void addTurnOnCurvesParams(std::string_view versionName, std::array<std::array<float, 10>, 14>& params);
  bool selectTurnOnCurvesParams(std::string_view versionName);

  float L0triggerProbability(float e, short ddl) const;

 private:
  static constexpr short NCHANNELS = 3136; ///< Number of trigger channels
  std::bitset<NCHANNELS> mTrigger2x2Map;   ///< Container for bad trigger cells, 1 means bad sell
  std::bitset<NCHANNELS> mTrigger4x4Map;   ///< Container for bad trigger cells, 1 means bad sell

  short mVersion;                       //current parameterization of turn-on curves
  static constexpr short NDDL = 14;     ///< Non-existing channels 56*64*1.5+1
  static constexpr short NMAXPAR = 10;  ///< Non-existing channels 56*64*1.5+1
  std::vector<std::string> mParamDescr; ///< Names of available parameterizations
  std::vector<std::array<std::array<float, NMAXPAR>, NDDL>> mParamSets;
  std::array<std::array<float, NMAXPAR>, NDDL> mCurrentSet;

  ClassDefNV(TriggerMap, 1);
};

} // namespace phos

} // namespace o2
#endif
