// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Param.h
/// \brief Parameters used for TPC space point calibration
///
/// \author Ole Schmidt, ole.schmidt@cern.ch

#ifndef ALICEO2_TPC_PARAM_H_
#define ALICEO2_TPC_PARAM_H_

#include "DataFormatsTPC/Constants.h"

#define TPC_RUN2

namespace o2
{
namespace TPC
{
namespace param
{
#ifdef TPC_RUN2
/// TPC geometric constants for Run 1+2
static constexpr int NPadRows = 159;                                   ///< total number of TPC pad rows
static constexpr int NROCTypes = 3;                                    ///< how many different pitches we have between the pad rows
static constexpr int NRowsPerROC[NROCTypes] = { 63, 64, 32 };          ///< number of rows for the different pitches
static constexpr int NRowsAccumulated[NROCTypes] = { 63, 127, 159 };   ///< accumulate number of rows (only used as abbreviation)
static constexpr float MinX[NROCTypes] = { 85.225f, 135.1f, 199.35f }; ///< x-position of first row for each ROC type
static constexpr float RowDX[NROCTypes] = { .75f, 1.f, 1.5f };         ///< row pitches
static constexpr float ROCDX[NROCTypes - 1] = { 3.375f, 1.25f };       ///< radial distance between the different ROCs
static constexpr float MaxX = 246.f;                                   ///< maximum radius for the TPC
#else                                                                  // not defined TPC_RUN2
/// TPC geometric constants for Run 3+
static constexpr int NPadRows = o2::TPC::Constants::MAXGLOBALPADROW;
static constexpr int NROCTypes = 4;
static constexpr int NRowsPerROC[NROCTypes] = { 63, 34, 30, 25 };
static constexpr int NRowsAccumulated[NROCTypes] = { 63, 97, 127, 152 };
static constexpr float MinX[NROCTypes] = { 85.225f, 135.2f, 171.4f, 209.65f };
static constexpr float RowDX[NROCTypes] = { .75f, 1.f, 1.2f, 1.5f };
static constexpr float ROCDX[NROCTypes - 1] = { 3.475f, 3.2f, 3.45f };
static constexpr float MaxX = 247.15f;
#endif                                                                 // defined TPC_RUN2
} // namespace param
} // namespace TPC
} // namespace o2
#endif
