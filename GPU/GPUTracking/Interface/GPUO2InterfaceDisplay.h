// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2InterfaceDisplay.h
/// \author David Rohr

#ifndef GPUO2INTERFACEDisplay_H
#define GPUO2INTERFACEDisplay_H

// Some defines denoting that we are compiling for O2
#ifndef GPUCA_HAVE_O2HEADERS
#define GPUCA_HAVE_O2HEADERS
#endif
#ifndef GPUCA_TPC_GEOMETRY_O2
#define GPUCA_TPC_GEOMETRY_O2
#endif
#ifndef GPUCA_O2_INTERFACE
#define GPUCA_O2_INTERFACE
#endif

#include <memory>
#include <vector>

namespace o2::gpu
{
class GPUDisplay;
class GPUQA;
struct GPUParam;
struct GPUTrackingInOutPointers;
struct GPUO2InterfaceConfiguration;
class GPUDisplayBackend;
class GPUO2InterfaceDisplay
{
 public:
  GPUO2InterfaceDisplay(const GPUO2InterfaceConfiguration* config = nullptr);
  ~GPUO2InterfaceDisplay();

  int startDisplay();
  int show(const GPUTrackingInOutPointers* ptrs);
  int endDisplay();

 private:
  std::unique_ptr<GPUDisplay> mDisplay;
  std::unique_ptr<GPUQA> mQA;
  std::unique_ptr<GPUParam> mParam;
  std::unique_ptr<GPUDisplayBackend> mBackend;
  std::unique_ptr<GPUO2InterfaceConfiguration> mConfig;
};
} // namespace o2::gpu

#endif
