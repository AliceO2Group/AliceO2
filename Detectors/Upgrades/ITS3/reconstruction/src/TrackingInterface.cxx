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

#include "ITS3Reconstruction/TrackingInterface.h"
#include "ITS3Reconstruction/IOUtils.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsBase/GRPGeomHelper.h"

namespace o2::its3
{

void ITS3TrackingInterface::updateTimeDependentParams(framework::ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::its3::TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
    pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("alppar");
    if (pc.inputs().getPos("itsTGeo") >= 0) {
      pc.inputs().get<o2::its::GeometryTGeo*>("itsTGeo");
    }
    auto geom = its::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
    getConfiguration(pc);
  }
}

void ITS3TrackingInterface::finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == framework::ConcreteDataMatcher("IT3", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    setClusterDictionary((const o2::its3::TopologyDictionary*)obj);
    return;
  }
  if (matcher == framework::ConcreteDataMatcher("ITS", "ALPIDEPARAM", 0)) {
    LOG(info) << "Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    par.printKeyValues();
    return;
  }
  if (matcher == framework::ConcreteDataMatcher("GLO", "MEANVERTEX", 0)) {
    LOGP(info, "Mean vertex acquired");
    setMeanVertex((const o2::dataformats::MeanVertexObject*)obj);
    return;
  }
  if (matcher == framework::ConcreteDataMatcher("ITS", "GEOMTGEO", 0)) {
    LOG(info) << "ITS GeometryTGeo loaded from ccdb";
    o2::its::GeometryTGeo::adopt((o2::its::GeometryTGeo*)obj);
    return;
  }
}

void ITS3TrackingInterface::loadROF(gsl::span<itsmft::ROFRecord>& trackROFspan,
                                    gsl::span<const itsmft::CompClusterExt> clusters,
                                    gsl::span<const unsigned char>::iterator& pattIt,
                                    const dataformats::MCTruthContainer<MCCompLabel>* mcLabels)
{
  ioutils::loadROFrameDataITS3(mTimeFrame, trackROFspan, clusters, pattIt, mDict, mcLabels);
}

} // namespace o2::its3
