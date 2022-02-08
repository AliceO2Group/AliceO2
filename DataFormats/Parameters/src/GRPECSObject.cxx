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
#include "CommonUtils/NameConf.h"

using namespace o2::parameters;
using o2::detectors::DetID;

//_______________________________________________
void GRPECSObject::print() const
{
  // print itself
  std::time_t ts = mTimeStart / 1000, te = mTimeEnd / 1000;
  printf("Run %d of type %d, period %s\n", mRun, int(mRunType), mDataPeriod.c_str());
  printf("Start: %s", std::ctime(&ts));
  printf("End  : %s", std::ctime(&te));
  printf("isMC : %d ", isMC());
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
GRPECSObject* GRPECSObject::loadFrom(const std::string& grpecsFileName)
{
  // load object from file
  auto fname = o2::base::NameConf::getGRPFileName(grpecsFileName);
  TFile flGRP(fname.c_str());
  if (flGRP.IsZombie()) {
    LOG(error) << "Failed to open " << fname;
    throw std::runtime_error("Failed to open GRPECS file");
  }
  auto grp = reinterpret_cast<o2::parameters::GRPECSObject*>(flGRP.GetObjectChecked(o2::base::NameConf::CCDBOBJECT.data(), Class()));
  if (!grp) {
    throw std::runtime_error(fmt::format("Failed to load GRPECS object from {}", fname));
  }
  return grp;
}
