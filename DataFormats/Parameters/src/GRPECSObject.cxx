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

/// \file GRPECSObject.cxx
/// \brief Implementation of General Run Parameters object from ECS
/// \author ruben.shahoyan@cern.ch

#include <FairLogger.h>
#include <TFile.h>
#include "DataFormatsParameters/GRPECSObject.h"
#include <cmath>
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::parameters;
using o2::detectors::DetID;

//_______________________________________________
void GRPECSObject::print() const
{
  // print itself
  std::time_t t = mTimeStart; // system_clock::to_time_t(mTimeStart);
  printf("Start: %s", std::ctime(&t));
  printf("Detectors: Cont.RO Triggers\n");
  for (auto i = DetID::First; i <= DetID::Last; i++) {
    if (!isDetReadOut(DetID(i))) {
      continue;
    }
    printf("%9s: ", DetID(i).getName());
    printf("%7s ", isDetContinuousReadOut(DetID(i)) ? "   +   " : "   -   ");
    printf("%7s ", isDetTriggers(DetID(i)) ? "   +   " : "   -   ");
    printf("\n");
  }
}

//_______________________________________________
void GRPECSObject::setDetROMode(o2::detectors::DetID id, ROMode status)
{
  /// set detector readout mode status
  if (!(status & PRESENT)) {
    remDetReadOut(id);
    return;
  }
  addDetReadOut(id);
  if ((status & CONTINUOUS) == CONTINUOUS) {
    addDetContinuousReadOut(id);
  } else {
    remDetContinuousReadOut(id);
  }
  if ((status & TRIGGERING) == TRIGGERING) {
    addDetTrigger(id);
  } else {
    remDetTrigger(id);
  }
}

//_______________________________________________
GRPECSObject::ROMode GRPECSObject::getDetROMode(o2::detectors::DetID id) const
{
  GRPECSObject::ROMode status = ABSENT;
  if (isDetReadOut(id)) {
    status = PRESENT;
  } else {
    if (isDetContinuousReadOut(id)) {
      status = GRPECSObject::ROMode(status | CONTINUOUS);
    }
    if (isDetTriggers(id)) {
      status = GRPECSObject::ROMode(status | TRIGGERING);
    }
  }
  return status;
}

//_______________________________________________
GRPECSObject* GRPECSObject::loadFrom(const std::string& grpecsFileName, const std::string& grpecsName)
{
  // load object from file
  auto fname = o2::base::NameConf::getGRPECSFileName(grpecsFileName);
  TFile flGRPECS(fname.c_str());
  if (flGRPECS.IsZombie()) {
    LOG(ERROR) << "Failed to open " << fname;
    throw std::runtime_error("Failed to open GRP ECS file");
  }
  auto grpecs = reinterpret_cast<o2::parameters::GRPECSObject*>(
    flGRPECS.GetObjectChecked(grpecsName.data(), o2::parameters::GRPECSObject::Class()));
  if (!grpecs) {
    LOG(ERROR) << "Did not find GRP ECS object named " << grpecsName;
    throw std::runtime_error("Failed to load GRP ECS object");
  }
  return grpecs;
}
