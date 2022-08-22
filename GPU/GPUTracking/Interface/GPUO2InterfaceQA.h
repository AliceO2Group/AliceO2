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

/// \file GPUO2InterfaceQA.h
/// \author David Rohr

#ifndef GPUO2INTERFACEQA_H
#define GPUO2INTERFACEQA_H

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

class TH1F;
class TH1D;
class TH2F;
class TObjArray;

namespace o2
{
class MCCompLabel;
namespace tpc
{
class TrackTPC;
struct ClusterNativeAccess;
} // namespace tpc
} // namespace o2

namespace o2::gpu
{
class GPUQA;
struct GPUParam;
struct GPUO2InterfaceConfiguration;
struct GPUSettingsGRP;
class GPUO2InterfaceQA
{
 public:
  GPUO2InterfaceQA(const GPUO2InterfaceConfiguration* config = nullptr);
  ~GPUO2InterfaceQA();

  int initializeForProcessing(int tasks); // only needed for processing, not for postprocessing

  void runQA(const std::vector<o2::tpc::TrackTPC>* tracksExternal, const std::vector<o2::MCCompLabel>* tracksExtMC, const o2::tpc::ClusterNativeAccess* clNative);
  int postprocess(TObjArray& out);
  void updateGRP(GPUSettingsGRP* grp);

  // Input might be modified, so we assume non-const. If it is const, a copy should be created before.
  int postprocessExternal(std::vector<TH1F>& in1, std::vector<TH2F>& in2, std::vector<TH1D>& in3, TObjArray& out, int tasks);

  void getHists(const std::vector<TH1F>*& h1, const std::vector<TH2F>*& h2, const std::vector<TH1D>*& h3);
  void resetHists();
  void cleanup();

 private:
  std::unique_ptr<GPUQA> mQA;
  std::unique_ptr<GPUParam> mParam;
};
} // namespace o2::gpu

#endif
