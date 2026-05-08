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

#include <string>
#include <format>

#include "ITS3Align/AlignmentTypes.h"
ClassImp(o2::its3::align::Point);
ClassImp(o2::its3::align::FrameInfoExt);
ClassImp(o2::its3::align::FitInfo);
ClassImp(o2::its3::align::Track);

std::string o2::its3::align::FrameInfoExt::asString() const
{
  return std::format("Sensor={} Layer={} X={} Alpha={}\n\tMEAS: y={} z={}", sens, lr, x, alpha, positionTrackingFrame[0], positionTrackingFrame[1]);
}
