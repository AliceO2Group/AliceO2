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

#include "CommonDataFormat/TimeStamp.h"

template class o2::dataformats::TimeStamp<int>;
template class o2::dataformats::TimeStamp<float>;
template class o2::dataformats::TimeStamp<double>;
template class o2::dataformats::TimeStampWithError<int, int>;
template class o2::dataformats::TimeStampWithError<float, float>;
template class o2::dataformats::TimeStampWithError<double, double>;
