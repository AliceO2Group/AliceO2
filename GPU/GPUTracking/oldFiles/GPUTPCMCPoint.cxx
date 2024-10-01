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

/// \file GPUTPCMCPoint.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCMCPoint.h"

GPUTPCMCPoint::GPUTPCMCPoint() : fX(0), fY(0), fZ(0), fSx(0), fSy(0), fSz(0), fTime(0), mISlice(0), fTrackID(0)
{
  //* Default constructor
}
