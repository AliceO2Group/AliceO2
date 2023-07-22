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

#include "FT0Raw/DigitBlockFT0.h"

template class o2::fit::DigitBlockFIT<o2::ft0::SingleLUT, o2::ft0::Digit, o2::ft0::ChannelData>;
template class o2::fit::DigitBlockFIText<o2::ft0::SingleLUT, o2::ft0::Digit, o2::ft0::ChannelData, o2::ft0::TriggersExt>;