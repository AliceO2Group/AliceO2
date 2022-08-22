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

/// \file DetectorData.cxx
/// \author Piotr Nowakowski

#include "EveWorkflow/DetectorData.h"
#include "EveWorkflow/EveWorkflowHelper.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsParameters/GRPObject.h"
#include "CommonUtils/NameConf.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/ClustererParam.h"
#include "TRDBase/GeometryFlat.h"
#include "TOFBase/Geo.h"
#include "TRDBase/Geometry.h"
#include "MCHTracking/TrackExtrap.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::event_visualisation;
using namespace o2::framework;
using namespace o2::dataformats;
using namespace o2::globaltracking;
using namespace o2::tpc;
using namespace o2::trd;

void DetectorData::init()
{
  const auto grp = o2::base::GRPGeomHelper::instance().getGRPECS();
  mConfig.configGRP.solenoidBz = o2::base::Propagator::Instance()->getNominalBz();
  mConfig.configGRP.continuousMaxTimeBin = grp->isDetContinuousReadOut(o2::detectors::DetID::TPC) ? -1 : 0; // Number of timebins in timeframe if continuous, 0 otherwise
  mConfig.ReadConfigurableParam();

  auto gm = o2::trd::Geometry::instance();
  gm->createPadPlaneArray();
  gm->createClusterMatrixArray();
  mTrdGeo.reset(new o2::trd::GeometryFlat(*gm));
  mConfig.configCalib.trdGeometry = mTrdGeo.get();

  o2::tof::Geo::Init();

  o2::its::GeometryTGeo::Instance()->fillMatrixCache(
    o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot,
                             o2::math_utils::TransformType::T2G,
                             o2::math_utils::TransformType::L2G,
                             o2::math_utils::TransformType::T2L));

  o2::mch::TrackExtrap::setField();
}

void DetectorData::setITSDict(const o2::itsmft::TopologyDictionary* d)
{
  mITSDict = d;
  mConfig.configCalib.itsPatternDict = d;
}

void DetectorData::setMFTDict(const o2::itsmft::TopologyDictionary* d)
{
  mMFTDict = d;
  mConfig.configCalib.mftPatternDict = d;
}
