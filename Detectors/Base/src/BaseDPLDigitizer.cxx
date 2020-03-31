// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <DetectorsBase/BaseDPLDigitizer.h>
#include <SimConfig/DigiParams.h>
#include <DetectorsBase/GeometryManager.h>
#include <DataFormatsParameters/GRPObject.h>
#include <DetectorsBase/Propagator.h>
#include <FairLogger.h>

using namespace o2::base;

BaseDPLDigitizer::BaseDPLDigitizer(InitServices::Type servicecode)
{
  mNeedGeom = servicecode & InitServices::GEOM;
  mNeedField = servicecode & InitServices::FIELD;
}

void BaseDPLDigitizer::init(o2::framework::InitContext& ic)
{
  // init basic stuff when this was asked for
  if (mNeedGeom) {
    LOG(INFO) << "Initializing geometry service";
    o2::base::GeometryManager::loadGeometry(o2::conf::DigiParams::Instance().digitizationgeometry);
  }

  if (mNeedField) {
    LOG(INFO) << "Initializing field service";
    // load from GRP
    auto inputGRP = o2::conf::DigiParams::Instance().grpfile;
    if (inputGRP.empty()) {
      LOG(ERROR) << "GRP filename not initialized in DigiParams";
    }
    auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
    if (!grp) {
      LOG(ERROR) << "This workflow needs a valid GRP file to start";
    }
    // init magnetic field
    o2::base::Propagator::initFieldFromGRP(grp);
  }

  // finally call specific init
  this->initDigitizerTask(ic);
}
