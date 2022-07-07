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

#ifndef O2_TPCVDRIFTTGL_CALIB_DEVICE_H
#define O2_TPCVDRIFTTGL_CALIB_DEVICE_H

/// @file   TPCVDriftTglCalibratorSpec.h
/// @brief  Device to calibrate vdrift with tg_lambda

#include "Framework/DataProcessorSpec.h"

namespace o2::tpc
{

o2::framework::DataProcessorSpec getTPCVDriftTglCalibSpec(int ntgl, float tglMax, int ndtgl, float dtglMax, size_t slotL, size_t minEnt);
}

#endif
