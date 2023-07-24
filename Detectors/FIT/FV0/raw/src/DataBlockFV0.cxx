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

#include "FV0Raw/DataBlockFV0.h"

template class o2::fit::DataBlockPM<o2::fit::DataBlockConfig<false>, o2::fv0::RawHeaderPM, o2::fv0::RawDataPM>;
template class o2::fit::DataBlockTCM<o2::fit::DataBlockConfig<false>, o2::fv0::RawHeaderTCM, o2::fv0::RawDataTCM>;
template class o2::fit::DataBlockTCMext<o2::fit::DataBlockConfig<false>, o2::fv0::RawHeaderTCMext, o2::fv0::RawDataTCM, o2::fv0::RawDataTCMext>;
