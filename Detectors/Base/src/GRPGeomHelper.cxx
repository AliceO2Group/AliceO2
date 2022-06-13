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

/// \file GRPGeomHelper.cxx
/// \brief Helper for geometry and GRP related CCDB requests
/// \author ruben.shahoyan@cern.ch

#include <fmt/format.h>
#include <TGeoGlobalMagField.h>
#include <TGeoManager.h>
#include "Field/MagneticField.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/ProcessingContext.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputSpec.h"
#include "Framework/InputRecord.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DataFormatsParameters/GRPLHCIFData.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "DataFormatsParameters/GRPMagField.h"

using DetID = o2::detectors::DetID;
using namespace o2::base;
using namespace o2::framework;
namespace o2d = o2::dataformats;

GRPGeomRequest::GRPGeomRequest(bool orbitResetTime, bool GRPECS, bool GRPLHCIF, bool GRPMagField, bool askMatLUT, GeomRequest geom, std::vector<o2::framework::InputSpec>& inputs)
  : askGRPECS(GRPECS), askGRPLHCIF(GRPLHCIF), askGRPMagField(GRPMagField), askMatLUT(askMatLUT), askTime(orbitResetTime)
{
  if (geom == Aligned) {
    askGeomAlign = true;
    addInput({"geomAlp", "GLO", "GEOMALIGN", 0, Lifetime::Condition, ccdbParamSpec("GLO/Config/GeometryAligned")}, inputs);
  } else if (geom == Ideal || geom == Alignments) {
    askGeomIdeal = true;
    addInput({"geomIdeal", "GLO", "GEOMIDEAL", 0, Lifetime::Condition, ccdbParamSpec("GLO/Config/Geometry")}, inputs);
  }
  if (geom == Alignments) {
    askAlignments = true;
    for (auto id = DetID::First; id <= DetID::Last; id++) {
      std::string binding = fmt::format("align{}", DetID::getName(id));
      addInput({binding, DetID::getDataOrigin(id), "ALIGNMENT", 0, Lifetime::Condition, ccdbParamSpec(fmt::format("{}/Calib/Align", DetID::getName(id)))}, inputs);
    }
  }
  if (askMatLUT) {
    addInput({"matLUT", "GLO", "MATLUT", 0, Lifetime::Condition, ccdbParamSpec("GLO/Param/MatLUT")}, inputs);
  }
  if (askTime) {
    addInput({"orbitReset", "CTP", "ORBITRESET", 0, Lifetime::Condition, ccdbParamSpec("CTP/Calib/OrbitReset")}, inputs);
  }
  if (askGRPECS) {
    addInput({"grpecs", "GLO", "GRPECS", 0, Lifetime::Condition, ccdbParamSpec("GLO/Config/GRPECS")}, inputs);
  }
  if (askGRPLHCIF) {
    addInput({"grplhcif", "GLO", "GRPLHCIF", 0, Lifetime::Condition, ccdbParamSpec("GLO/Config/GRPLHCIF")}, inputs);
  }
  if (askGRPMagField) {
    addInput({"grpfield", "GLO", "GRPMAGFIELD", 0, Lifetime::Condition, ccdbParamSpec("GLO/Config/GRPMagField")}, inputs);
  }
}

void GRPGeomRequest::addInput(const o2::framework::InputSpec&& isp, std::vector<o2::framework::InputSpec>& inputs)
{
  if (std::find(inputs.begin(), inputs.end(), isp) == inputs.end()) {
    inputs.emplace_back(isp);
  }
}

//=====================================================================================

void GRPGeomHelper::setRequest(std::shared_ptr<GRPGeomRequest> req)
{
  if (mRequest) {
    LOG(fatal) << "GRP/Geometry CCDB request was already set";
  }
  mRequest = req;
}

void GRPGeomHelper::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (mRequest->askGRPMagField && matcher == ConcreteDataMatcher("GRP", "GRPMAGFIELD", 0)) {
    bool needInit = mGRPMagField == nullptr;
    mGRPMagField = (o2::parameters::GRPMagField*)obj;
    LOG(info) << "GRP MagField object updated";
    if (needInit) {
      o2::base::Propagator::initFieldFromGRP(mGRPMagField);
    } else {
      auto field = TGeoGlobalMagField::Instance()->GetField();
      if (field->InheritsFrom("o2::field::MagneticField")) {
        ((o2::field::MagneticField*)field)->rescaleField(mGRPMagField->getL3Current(), mGRPMagField->getDipoleCurrent(), mGRPMagField->getFieldUniformity());
      }
      o2::base::Propagator::Instance(false)->updateField();
    }
    return;
  }
  if (mRequest->askGRPECS && matcher == ConcreteDataMatcher("GRP", "GRPECS", 0)) {
    mGRPECS = (o2::parameters::GRPECSObject*)obj;
    LOG(info) << "GRP ECS object updated";
    return;
  }
  if (mRequest->askGRPLHCIF && matcher == ConcreteDataMatcher("GRP", "GRPLHCIF", 0)) {
    mGRPLHCIF = (o2::parameters::GRPLHCIFData*)obj;
    LOG(info) << "GRP LHCIF object updated";
    return;
  }
  if (mRequest->askTime && matcher == ConcreteDataMatcher("CTP", "ORBITRESET", 0)) {
    mOrbitResetTimeMS = (*(std::vector<Long64_t>*)obj)[0] / 1000;
    LOG(info) << "orbit reset time updated to " << mOrbitResetTimeMS;
    return;
  }
  if (mRequest->askMatLUT && matcher == ConcreteDataMatcher("GLO", "MATLUT", 0)) {
    LOG(info) << "material LUT updated";
    mMatLUT = o2::base::MatLayerCylSet::rectifyPtrFromFile((o2::base::MatLayerCylSet*)obj);
    o2::base::Propagator::Instance(false)->setMatLUT(mMatLUT);
    return;
  }
  if (mRequest->askGeomAlign && matcher == ConcreteDataMatcher("GLO", "GEOMALIGN", 0)) {
    LOG(info) << "aligned geometry updated";
    return;
  }
  if (mRequest->askGeomIdeal && matcher == ConcreteDataMatcher("GLO", "GEOMIDEAL", 0)) {
    LOG(info) << "ideal geometry updated";
    return;
  }
  constexpr o2::header::DataDescription algDesc{"ALIGNMENT"};
  if (mRequest->askAlignments && matcher.description == algDesc) {
    for (auto id = DetID::First; id <= DetID::Last; id++) {
      if (matcher.origin == DetID::getDataOrigin(id)) {
        LOG(info) << DetID::getName(id) << " alignment updated";
        mAlignments[id] = (std::vector<o2::detectors::AlignParam>*)obj;
        break;
      }
    }
    return;
  }
}

void GRPGeomHelper::checkUpdates(ProcessingContext& pc) const
{
  // request input just to trigger finaliseCCDB if there was an update
  if (mRequest->askGRPMagField) {
    pc.inputs().get<o2::parameters::GRPMagField*>("grpfield");
  }
  if (mRequest->askGRPLHCIF) {
    pc.inputs().get<o2::parameters::GRPLHCIFData*>("grplhcif");
  }
  if (mRequest->askGRPECS) {
    pc.inputs().get<o2::parameters::GRPECSObject*>("grpecs");
  }
  if (mRequest->askTime) {
    pc.inputs().get<std::vector<Long64_t>*>("orbitReset");
  }
  if (mRequest->askMatLUT) {
    pc.inputs().get<o2::base::MatLayerCylSet*>("matLUT");
  }
  if (mRequest->askGeomAlign) {
    pc.inputs().get<TGeoManager*>("geomAlp");
  } else if (mRequest->askGeomIdeal) {
    pc.inputs().get<TGeoManager*>("geomIdeal");
  }
  if (mRequest->askAlignments) {
    for (auto id = DetID::First; id <= DetID::Last; id++) {
      std::string binding = fmt::format("align{}", DetID::getName(id));
      pc.inputs().get<std::vector<o2::detectors::AlignParam>*>(binding);
    }
  }
}

int GRPGeomHelper::getNHBFPerTF()
{
  return instance().mGRPECS ? instance().mGRPECS->getNHBFPerTF() : 128;
}
