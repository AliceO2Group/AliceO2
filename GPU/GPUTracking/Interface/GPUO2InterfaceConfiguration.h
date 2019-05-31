// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2InterfaceConfiguration.h
/// \author David Rohr

#ifndef GPUO2INTERFACECONFIGURATION_H
#define GPUO2INTERFACECONFIGURATION_H

#ifndef GPUCA_O2_LIB
#define GPUCA_O2_LIB
#endif
#ifndef HAVE_O2HEADERS
#define HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif

#include <memory>
#include "GPUSettings.h"
#include "GPUDisplayConfig.h"
#include "GPUQAConfig.h"
#include "GPUDataTypes.h"

namespace o2
{
namespace base
{
class MatLayerCylSet;
}
namespace trd
{
class TRDGeometryFlat;
}
namespace gpu
{
class TPCFastTransform;
// Full configuration structure with all available settings of GPU...
struct GPUO2InterfaceConfiguration {
  GPUO2InterfaceConfiguration() = default;
  ~GPUO2InterfaceConfiguration() = default;
  GPUO2InterfaceConfiguration(const GPUO2InterfaceConfiguration&) = default;

  // Settings for the Interface class
  struct GPUInterfaceSettings {
    bool dumpEvents = false;
  };

  GPUSettingsProcessing configProcessing;
  GPUSettingsDeviceProcessing configDeviceProcessing;
  GPUSettingsEvent configEvent;
  GPUSettingsRec configReconstruction;
  GPUDisplayConfig configDisplay;
  GPUQAConfig configQA;
  GPUInterfaceSettings configInterface;
  GPURecoStepConfiguration configWorkflow;
  const TPCFastTransform* fastTransform = nullptr;
  const o2::base::MatLayerCylSet* matLUT = nullptr;
  const o2::trd::TRDGeometryFlat* trdGeometry = nullptr;
};
} // namespace gpu
} // namespace o2

#endif
