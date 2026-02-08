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

#ifndef ALICEO2_PASSIVE_EXTERNALMODULE_H
#define ALICEO2_PASSIVE_EXTERNALMODULE_H

#include "DetectorsPassive/PassiveBase.h" // base class of passive modules
#include "Rtypes.h"                       // for Pipe::Class, ClassDef, Pipe::Streamer

class TGeoVolume;
class TGeoTransformation;

namespace o2
{
namespace passive
{

// options used to configure a generic plug and play external module
struct ExternalModuleOptions {
  std::string root_macro_file;           // the file where to lookup the ROOT geometry building macro
  std::string top_volume;                // the volume to be added
  std::string anchor_volume;             // the volume into which the detector will be hooked
  TGeoMatrix const* placement = nullptr; // how to place the module inside anchor_volume
};

// a module (passive material) defined externally (ROOT macro / GDML / TGeo geometry)
class ExternalModule : public PassiveBase
{
 public:
  ExternalModule(const char* name, const char* long_title, ExternalModuleOptions options);
  ExternalModule() = default; // default constructor

  ~ExternalModule() override = default;
  void ConstructGeometry() override;

  /// Clone this object (used in MT mode only)
  FairModule* CloneModule() const override { return nullptr; }

  typedef std::function<TGeoVolume const*()> GeomBuilderFcn; // function hook for external geometry builder

 private:
  // void createMaterials();
  ExternalModule(const ExternalModule& orig);
  ExternalModule& operator=(const ExternalModule&);

  GeomBuilderFcn mGeomHook;
  ExternalModuleOptions mOptions;

  bool initGeomBuilderHook();       // function to load/JIT Geometry builder hook
  void remapMedia(TGeoVolume* vol); // performs a remapping of materials/media IDs after registration with VMC

  // ClassDefOverride(ExternalModule, 0);
};
} // namespace passive
} // namespace o2
#endif
