// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See http://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_LOCALGAINFACTOR_H
#define O2_TRD_LOCALGAINFACTOR_H

////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
//  TRD calibration class for parameters which are stored pad wise (1.2M entries) //
//  2019 - Ported from various bits of AliRoot (SHTM)                             //
//  originally in TRD run2 parlance, this would be a AliTRDCalPad instantiated    //
//  as a LocalT0                                                                  //
////////////////////////////////////////////////////////////////////////////////////

#include "TRDBase/PadParameters.h"
#include "TRDBase/PadCalibrations.h"
namespace o2
{
namespace trd
{

class LocalGainFactor : public PadCalibrations<float>
{
 public:
  using PadCalibrations<float>::PadCalibrations;
  ~LocalGainFactor() = default;
};
} // namespace trd
} // namespace o2
#endif
