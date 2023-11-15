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

/// @file   ZDCConstants.h
/// @brief  Some ZDC constants shared between O2 and O2Physics

/// @author pietro.cortese@cern.ch

#ifndef ALICEO2_ZDCCONSTANTS_H_
#define ALICEO2_ZDCCONSTANTS_H_

namespace o2
{
namespace zdc
{

//< map detector/tower to continuous channel Id
constexpr int IdDummy = -1;
constexpr int IdVoid = -2;

constexpr int IdZNAC = 0;
constexpr int IdZNA1 = 1;
constexpr int IdZNA2 = 2;
constexpr int IdZNA3 = 3;
constexpr int IdZNA4 = 4;
constexpr int IdZNASum = 5;
//
constexpr int IdZPAC = 6;
constexpr int IdZPA1 = 7;
constexpr int IdZPA2 = 8;
constexpr int IdZPA3 = 9;
constexpr int IdZPA4 = 10;
constexpr int IdZPASum = 11;
//
constexpr int IdZEM1 = 12;
constexpr int IdZEM2 = 13;
//
constexpr int IdZNCC = 14;
constexpr int IdZNC1 = 15;
constexpr int IdZNC2 = 16;
constexpr int IdZNC3 = 17;
constexpr int IdZNC4 = 18;
constexpr int IdZNCSum = 19;
//
constexpr int IdZPCC = 20;
constexpr int IdZPC1 = 21;
constexpr int IdZPC2 = 22;
constexpr int IdZPC3 = 23;
constexpr int IdZPC4 = 24;
constexpr int IdZPCSum = 25;

} // namespace zdc
} // namespace o2

#endif
