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

/// \file DigitTimebin.h
/// \brief EMCAL DigitTimebin for the DigitsWriteoutBuffer and DigitsWriteoutBufferTRU
#ifndef ALICEO2_EMCAL_DIGITTIMEBIN_H
#define ALICEO2_EMCAL_DIGITTIMEBIN_H

#include <array>
#include <gsl/span>
#include "Rtypes.h"
#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALSimulation/LabeledDigit.h"
#include "DataFormatsEMCAL/Cell.h"
#include "EMCALBase/Geometry.h"

namespace o2
{

namespace emcal
{

template <class DigitTemplate>
struct DigitTimebinTemplate;

/// \brief DigitTimebin is DigitTimebinTemplate<LabeledDigit>
// typedef DigitTimebinTemplate<LabeledDigit> DigitTimebin2;
using DigitTimebin = DigitTimebinTemplate<LabeledDigit>;
using DigitTimebinTRU = DigitTimebinTemplate<Digit>;

/// \brief DigitTimebinTRU is DigitTimebinTemplate<Digit>
// typedef DigitTimebinTemplate<Digit>
//   DigitTimebinTRU2;

/// \struct DigitTimebinTemplate
/// \brief DigitTimebinTemplate templated, used for the DigitsWriteoutBuffer and DigitsWriteoutBufferTRU
/// \ingroup EMCALsimulation
/// \author Markus Fasel, ORNL
/// \author Hadi Hassan, ORNL
/// \author Simone Ragoni, Creighton U.
/// \date 16/12/2022
///
/// \param mRecordMode record mode
/// \param mEndWindow end window
/// \param mTriggerColl trigger collision
/// \param mInterRecord InteractionRecord
/// \param mDigitMap map of the digits, templated
template <class DigitTemplate>
struct DigitTimebinTemplate {
  bool mRecordMode = false;
  bool mEndWindow = false;
  bool mTriggerColl = false;
  std::optional<o2::InteractionRecord> mInterRecord;
  std::shared_ptr<std::unordered_map<int, std::list<DigitTemplate>>> mDigitMap = std::make_shared<std::unordered_map<int, std::list<DigitTemplate>>>();
  ClassDefNV(DigitTimebinTemplate, 1);
};

} // namespace emcal
} // namespace o2
#endif /* ALICEO2_EMCAL_DIGITTIMEBIN_H */
