// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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
/// \file TrackingGeometryManager.cxx
/// \author Paolo Butti
///

#include "ACTSInterface/TrackingGeometryManager.h"

#include <filesystem>
#include <stdexcept>
#include <utility>

#include <TFile.h>
#include <TGeoManager.h>

#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "ActsPlugins/Json/JsonMaterialDecorator.hpp"
#include "ActsPlugins/Json/MaterialMapJsonConverter.hpp"

#include "CommonUtils/NameConf.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsCommonDataFormats/DetMatrixCache.h"
#include "Framework/Logger.h"

namespace o2::acts
{

namespace
{
/// Whether the file holds a geometry under one of the names
/// o2::base::GeometryManager::loadGeometry() accepts.
bool hasO2GeometryObject(std::string_view fileName)
{
  TFile file(std::string(fileName).c_str());
  if (file.IsZombie()) {
    throw std::runtime_error("o2::acts::TrackingGeometryManager: cannot open '" + std::string(fileName) + "'");
  }
  return file.GetKey(std::string(o2::base::NameConf::CCDBOBJECT).c_str()) != nullptr ||
         file.GetKey(std::string(o2::base::NameConf::GEOMOBJECTNAME_FAIR).c_str()) != nullptr;
}
} // namespace

TrackingGeometryManager::TrackingGeometryManager()
  : mNominalContext(Acts::GeometryContext::dangerouslyDefaultConstruct())
{
}

TrackingGeometryManager& TrackingGeometryManager::instance()
{
  static TrackingGeometryManager inst;
  return inst;
}

void TrackingGeometryManager::setBuilder(std::unique_ptr<ITrackingGeometryBuilder> builder)
{
  if (isBuilt()) {
    throw std::runtime_error(
      "o2::acts::TrackingGeometryManager: cannot change the builder after the geometry was built. "
      "Call clear() first if this is really intended.");
  }
  mBuilder = std::move(builder);
}

void TrackingGeometryManager::clear()
{
  // Order matters: the surfaces hold non-owning back-pointers into the element
  // store, so the geometry must go first.
  mGeometry.reset();
  mElementStore.clear();
  mIndex.clear();
  mDetIndices.clear();
  mNDecoratedSurfaces = 0;
}

void TrackingGeometryManager::build()
{
  if (isBuilt()) {
    LOG(info) << "ACTS tracking geometry is already built, nothing to do";
    return;
  }
  if (!o2::base::GeometryManager::isGeometryLoaded()) {
    throw std::runtime_error(
      "o2::acts::TrackingGeometryManager: no TGeo geometry is loaded. Declare "
      "o2::base::GRPGeomRequest::Aligned in the workflow's GRPGeomRequest and call "
      "GRPGeomHelper::instance().checkUpdates(pc) before asking for the ACTS geometry, or use "
      "buildFromFile() outside a workflow.");
  }
  doBuild();
}

void TrackingGeometryManager::buildFromFile(std::string_view geomFileName)
{
  if (isBuilt()) {
    LOG(info) << "ACTS tracking geometry is already built, nothing to do";
    return;
  }
  if (o2::base::GeometryManager::isGeometryLoaded()) {
    LOG(warn) << "A TGeo geometry is already loaded, ignoring the requested file and using it";
    build();
    return;
  }

  LOGP(info, "Loading TGeo geometry from '{}'", geomFileName);
  if (hasO2GeometryObject(geomFileName)) {
    o2::base::GeometryManager::loadGeometry(geomFileName);
  } else {
    // Not an o2-sim / CCDB geometry file. GeometryManager::loadGeometry() only
    // accepts those two object names and aborts on anything else, so fall back to
    // a plain import; this is the path for geometries exported from GDML, which
    // the standalone actsO2 chain uses.
    LOGP(warn, "'{}' holds no '{}' or '{}' object: importing it as a plain TGeo file",
         geomFileName, o2::base::NameConf::CCDBOBJECT, o2::base::NameConf::GEOMOBJECTNAME_FAIR);
    if (TGeoManager::Import(std::string(geomFileName).c_str()) == nullptr) {
      throw std::runtime_error("o2::acts::TrackingGeometryManager: cannot import TGeo geometry from '" +
                               std::string(geomFileName) + "'");
    }
  }
  build();
}

void TrackingGeometryManager::doBuild()
{
  if (mBuilder == nullptr) {
    throw std::runtime_error(
      "o2::acts::TrackingGeometryManager: no builder installed. Call setBuilder() with the "
      "detector-specific ITrackingGeometryBuilder before requesting the geometry.");
  }

  LOGP(info, "Building the ACTS tracking geometry with the '{}' builder", mBuilder->getName());
  auto output = mBuilder->build(*gGeoManager, mNominalContext);
  if (output.geometry == nullptr) {
    throw std::runtime_error(std::string("o2::acts::TrackingGeometryManager: builder '") +
                             mBuilder->getName() + "' returned no geometry");
  }

  mElementStore = std::move(output.elementStore);
  mIndex = std::move(output.index);

  if (!mMaterialMapFile.empty()) {
    // Must happen while the geometry is still non-const: Gen3 construction takes
    // no IMaterialDecorator, so material can only be injected afterwards.
    applyMaterialMapImpl(*output.geometry);
  }

  mGeometry = std::const_pointer_cast<const Acts::TrackingGeometry>(output.geometry);

  LOGP(info, "ACTS tracking geometry ready: {} mapped sensors, {} surfaces with material",
       mIndex.size(), mNDecoratedSurfaces);
}

void TrackingGeometryManager::applyMaterialMapImpl(Acts::TrackingGeometry& geometry)
{
  if (!std::filesystem::exists(mMaterialMapFile)) {
    throw std::runtime_error("o2::acts::TrackingGeometryManager: material map '" + mMaterialMapFile +
                             "' does not exist");
  }

  Acts::MaterialMapJsonConverter::Config converterConfig;
  converterConfig.context = mNominalContext;
  const Acts::JsonMaterialDecorator decorator(converterConfig, mMaterialMapFile, mLogLevel);

  mNDecoratedSurfaces = 0;
  geometry.apply([&](Acts::Surface& surface) {
    decorator.decorate(surface);
    if (surface.surfaceMaterial() != nullptr) {
      ++mNDecoratedSurfaces;
    }
  });

  if (mNDecoratedSurfaces == 0) {
    // Material maps are keyed on Acts::GeometryIdentifier, which is assigned during
    // construction. Any structural change to the builder shifts those ids and would
    // otherwise silently yield a material-free geometry.
    throw std::runtime_error(
      "o2::acts::TrackingGeometryManager: material map '" + mMaterialMapFile +
      "' matched no surface. It was produced for a different geometry structure and must be "
      "regenerated for this one.");
  }
}

std::shared_ptr<const Acts::TrackingGeometry> TrackingGeometryManager::getShared()
{
  if (!isBuilt()) {
    build();
  }
  return mGeometry;
}

const Acts::TrackingGeometry& TrackingGeometryManager::get()
{
  return *getShared();
}

const SurfaceIndexMap& TrackingGeometryManager::getIndex()
{
  if (!isBuilt()) {
    build();
  }
  return mIndex;
}

const SurfaceIndexMap& TrackingGeometryManager::getIndex(const o2::detectors::DetMatrixCache& cache,
                                                         double toleranceCm,
                                                         SensorPathProvider pathProvider)
{
  if (!isBuilt()) {
    build();
  }
  const int detID = cache.getDetID();
  const auto it = mDetIndices.find(detID);
  if (it != mDetIndices.end()) {
    return it->second;
  }
  auto [inserted, ok] = mDetIndices.emplace(
    detID, makeSurfaceIndex(*mGeometry, mNominalContext, cache, toleranceCm, std::move(pathProvider)));
  return inserted->second;
}

} // namespace o2::acts
