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

///
/// \file   Geo.h
/// \author Antonio Franco - INFN Bari
/// \version 1.1
/// \date 15/02/2021

#include "HMPIDBase/Geo.h"
#include "HMPIDBase/Param.h"
#include "TGeoManager.h"
#include "TMath.h"
#include <fairlogger/Logger.h>
#include "DetectorsBase/GeometryManager.h"

ClassImp(o2::hmpid::Geo);

using namespace o2::hmpid;

//constexpr Bool_t Geo::FEAWITHMASKS[NSECTORS];

// ============= Geo Class implementation =======

/// Init :
void Geo::Init()
{
  LOG(info) << "hmpid::Geo: Initialization of HMPID parameters";
}

