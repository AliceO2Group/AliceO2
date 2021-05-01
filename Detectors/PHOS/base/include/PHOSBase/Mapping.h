// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class Mapping
/// \brief Checks validity of hardware address (HW) and transform it to digit AbsId index
///
/// \author Dmitri Peresunko
/// \since Jan.2020
///

#ifndef PHOSMAPPING_H_
#define PHOSMAPPING_H_

#include <string_view>
#include <vector>
#include <utility>
#include "Rtypes.h"

namespace o2
{

namespace phos
{

class Mapping
{
 public:
  enum ErrorStatus { kOK,
                     kWrongDDL,
                     kWrongHWAddress,
                     kWrongAbsId,
                     kWrongCaloFlag,
                     kNotInitialized };
  static constexpr short NCHANNELS = 14337;               ///< Number of channels starting from 1
  static constexpr short NHWPERDDL = 2048;                ///< Number of HW addressed per DDL
  static constexpr short NMaxHWAddress = 3929;            ///< Maximal HW address (size of array)
  static constexpr short NDDL = 14;                       ///< Total number of DDLs
  static constexpr short NTRUBranchReadoutChannels = 112; ///< Number of TRU readout channels per branch
  static constexpr short NTRUReadoutChannels = 3136;      ///< Total number of TRU readout channels
  static constexpr short TRUFinalProductionChannel = 123; // The last channel of production bits, contains markesr to choose between 2x2 and 4x4 algorithm

  enum CaloFlag { kHighGain,
                  kLowGain,
                  kTRU };

  ~Mapping() = default;

  //Getters for unique instance of Mapping
  static Mapping* Instance();
  static Mapping* Instance(std::basic_string_view<char> path);

  /// \brief convert hardware address to absId and caloFlag
  ErrorStatus hwToAbsId(short ddl, short hw, short& absId, CaloFlag& caloFlag) const;
  /// \brief convert absId and caloflag to hardware address and ddl
  ErrorStatus absIdTohw(short absId, short caloFlag, short& ddl, short& hwAddr) const;

  /// \brief convert ddl number to crorc and link number (TODO!!!)
  static void ddlToCrorcLink(short iddl, short& crorc, short& link)
  {
    crorc = iddl / 8;
    link = iddl % 8;
  }

  ErrorStatus setMapping();

  //Select TRU readout channels or TRU flag channels
  static bool isTRUReadoutchannel(short hwAddress) { return (hwAddress < 112) || (hwAddress > 2048 && hwAddress < 2048 + 112); }

 protected:
  Mapping() = default;
  Mapping(std::basic_string_view<char> path);

  /// \brief Construct vector for conversion only if necessary
  ErrorStatus constructAbsToHWMatrix();

 private:
  static Mapping* sMapping;                         ///< Pointer to the unique instance of the singleton
  std::string mPath = "";                           ///< path to mapping files
  bool mInitialized = false;                        ///< If conversion tables created
  bool mInvInitialized = false;                     ///< If inverse conversion tables created
  short mAbsId[NDDL][NMaxHWAddress] = {0};          ///< Conversion table (ddl,branch,fec,chip,channel) to absId
  CaloFlag mCaloFlag[NDDL][NMaxHWAddress] = {kTRU}; ///< Conversion table (ddl,branch,fec,chip,channel) to absId
  short mAbsToHW[NCHANNELS][3][2] = {0};            ///< Conversion table (AbsId,caloFlag) to pair (ddl, hw address)

  ClassDefNV(Mapping, 1);
}; // End of Mapping

} // namespace phos

} // namespace o2
#endif
