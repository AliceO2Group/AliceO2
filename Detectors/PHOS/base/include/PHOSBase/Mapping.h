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
  static constexpr short NCHANNELS = 14336;               ///< Number of channels starting from 1
  static constexpr short NHWPERDDL = 2048;                ///< Number of HW addressed per DDL
  static constexpr short NMaxHWAddress = 3929;            ///< Maximal HW address (size of array)
  static constexpr short NDDL = 14;                       ///< Total number of DDLs
  static constexpr short NTRUBranchReadoutChannels = 112; ///< Number of TRU readout channels per branch
  static constexpr short NTRUReadoutChannels = 3136;      ///< Total number of TRU readout channels
  static constexpr short TRUFinalProductionChannel = 123; // The last channel of production bits, contains markesr to choose between 2x2 and 4x4 algorithm

  enum CaloFlag { kLowGain,
                  kHighGain,
                  kTRU };

  ~Mapping() = default;

  // Getters for unique instance of Mapping
  static Mapping* Instance();
  static Mapping* Instance(std::basic_string_view<char> path);

  /// \brief convert hardware address to absId and caloFlag
  ErrorStatus hwToAbsId(short ddl, short hw, short& absId, CaloFlag& caloFlag) const;
  /// \brief convert absId and caloflag to hardware address and ddl
  ErrorStatus absIdTohw(short absId, short caloFlag, short& ddl, short& hwAddr) const;

  /// \brief convert ddl number to crorc and link number
  static void ddlToCrorcLink(short iddl, short& flp, short& crorc, short& link)
  {
    //     FLP164:
    // CRORC S/N 0243
    //       channel 0 -> human name M2-0, DDL 3
    //       channel 1 -> human name M2-1, DDL 4
    //       channel 2 -> human name M2-2, DDL 5
    //       channel 3 -> human name M2-3, DDL 6
    // CRORC S/N 0304
    //       channel 0 -> human name M1-2, DDL 1
    //       channel 1 -> human name M1-3, DDL 2

    // FLP165:
    // CRORC S/N 0106
    //       channel 0 -> human name M4-0, DDL 11
    //       channel 1 -> human name M4-1, DDL 12
    //       channel 2 -> human name M4-2, DDL 13
    //       channel 3 -> human name M4-3, DDL 14
    // CRORC S/N 0075
    //       channel 0 -> human name M3-0, DDL 7
    //       channel 1 -> human name M3-1, DDL 8
    //       channel 2 -> human name M3-2, DDL 9
    //       channel 3 -> human name M3-3, DDL 10
    if (iddl < 6) {
      flp = 164;
      if (iddl < 2) {
        crorc = 304;
        link = iddl;
      } else {
        crorc = 243;
        link = iddl - 2;
      }
    } else {
      flp = 165;
      if (iddl < 10) {
        crorc = 75;
      } else {
        crorc = 106;
      }
      link = (iddl - 6) % 4;
    }
  }

  ErrorStatus setMapping();

  // Select TRU readout channels or TRU flag channels
  static bool isTRUReadoutchannel(short hwAddress) { return (hwAddress < 112) || (hwAddress >= 2048 && hwAddress < 2048 + 112); }

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
