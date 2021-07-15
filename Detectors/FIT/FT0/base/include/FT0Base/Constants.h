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

/// \file   Constants.h
/// \brief  General constants in FT0
///
/// \author Artur Furs, afurs@cern.ch

#ifndef ALICEO2_FT0_CONSTANTS_
#define ALICEO2_FT0_CONSTANTS_

namespace o2
{
namespace ft0
{

struct Constants {
  constexpr static std::size_t sNPM = 18;                             //Number of PMs
  constexpr static std::size_t sNPM_LCS = 1;                          //Number of PM-LCSs
  constexpr static std::size_t sNTCM = 1;                             //Number of TCMs
  constexpr static std::size_t sNTOTAL_PM = sNPM + sNPM_LCS;          //Total number of PMs(PM + PM_LCS)
  constexpr static std::size_t sNTOTAL_FEE = sNPM + sNPM_LCS + sNTCM; //Total number of FEE modules

  constexpr static std::size_t sNCHANNELS_PER_PM = 12;                                  //Number of local channels per PM
  constexpr static std::size_t sNCHANNELS_PM = sNPM * sNCHANNELS_PER_PM;                //Number of PM(not LCS) channels
  constexpr static std::size_t sNCHANNELS_PM_LCS = sNPM_LCS * sNCHANNELS_PER_PM;        //Number of PM_LCS channels
  constexpr static std::size_t sNTOTAL_CHANNELS_PM = sNCHANNELS_PM + sNCHANNELS_PM_LCS; //Total number of PM(+LCS) channels
};

} // namespace ft0
} // namespace o2
#endif
