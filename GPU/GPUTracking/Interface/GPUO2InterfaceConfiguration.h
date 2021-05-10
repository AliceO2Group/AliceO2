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

#ifndef HAVE_O2HEADERS
#define HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif
#ifndef GPUCA_O2_INTERFACE
#define GPUCA_O2_INTERFACE
#endif

#include <memory>
#include <array>
#include <vector>
#include <functional>
#include <gsl/gsl>
#include "GPUSettings.h"
#include "GPUDataTypes.h"
#include "GPUHostDataTypes.h"
#include "GPUOutputControl.h"
#include "DataFormatsTPC/Constants.h"

class TH1F;
class TH1D;
class TH2F;

namespace o2
{
namespace tpc
{
class TrackTPC;
class Digit;
} // namespace tpc
namespace gpu
{
class TPCFastTransform;
class GPUReconstruction;
struct GPUSettingsO2;

struct GPUInterfaceQAOutputs {
  const std::vector<TH1F>* hist1;
  const std::vector<TH2F>* hist2;
  const std::vector<TH1D>* hist3;
};

struct GPUInterfaceOutputs : public GPUTrackingOutputs {
  GPUInterfaceQAOutputs qa;
};

// Full configuration structure with all available settings of GPU...
struct GPUO2InterfaceConfiguration {
  GPUO2InterfaceConfiguration() = default;
  ~GPUO2InterfaceConfiguration() = default;
  GPUO2InterfaceConfiguration(const GPUO2InterfaceConfiguration&) = default;

  // Settings for the Interface class
  struct GPUInterfaceSettings {
    int dumpEvents = 0;
    bool outputToExternalBuffers = false;
    float memoryBufferScaleFactor = 1.f;
    // These constants affect GPU memory allocation only and do not limit the CPU processing
    unsigned long maxTPCZS = 8192ul * 1024 * 1024;
    unsigned int maxTPCHits = 1024 * 1024 * 1024;
    unsigned int maxTRDTracklets = 128 * 1024;
    unsigned int maxITSTracks = 96 * 1024;
  };

  GPUSettingsDeviceBackend configDeviceBackend;
  GPUSettingsProcessing configProcessing;
  GPUSettingsGRP configGRP;
  GPUSettingsRec configReconstruction;
  GPUSettingsDisplay configDisplay;
  GPUSettingsQA configQA;
  GPUInterfaceSettings configInterface;
  GPURecoStepConfiguration configWorkflow;
  GPUCalibObjectsConst configCalib;

  GPUSettingsO2 ReadConfigurableParam();

 private:
  friend class GPUReconstruction;
  GPUSettingsO2 ReadConfigurableParam_internal();
};

} // namespace gpu
} // namespace o2

#endif
