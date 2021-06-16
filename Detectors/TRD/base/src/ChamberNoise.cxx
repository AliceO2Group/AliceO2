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

//                                                                           //
//  TRD calibration class for parameters which are saved frequently(/run)    //
//  2019 - Ported from various bits of AliRoot (SHTM)                        //
//  Most things were stored in AliTRDcalROC,AliTRDcalPad, AliTRDcalDet       //
///////////////////////////////////////////////////////////////////////////////

#include <array>

#include "TRDBase/Geometry.h"
#include "TRDBase/ChamberNoise.h"
#include "TH2D.h"

using namespace o2::trd;
