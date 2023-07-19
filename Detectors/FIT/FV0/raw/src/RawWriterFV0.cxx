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

#include "FV0Raw/RawWriterFV0.h"

template class o2::fit::RawWriterFIT<o2::fv0::DigitBlockFV0, o2::fv0::DataBlockPM, o2::fv0::DataBlockTCM>;
template class o2::fit::RawWriterFIT<o2::fv0::DigitBlockFV0, o2::fv0::DataBlockPM::DataBlockInvertedPadding_t, o2::fv0::DataBlockTCM::DataBlockInvertedPadding_t>;
