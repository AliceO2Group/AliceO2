// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2InterfaceQA.h
/// \author David Rohr

#ifndef GPUO2INTERFACEQA_H
#define GPUO2INTERFACEQA_H

// Some defines denoting that we are compiling for O2
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
#include <vector>

class TH1F;
class TH1D;
class TH2F;
class TObjArray;

namespace o2::gpu
{
class GPUQA;
class GPUSettingsQA;
class GPUO2InterfaceQA
{
 public:
  GPUO2InterfaceQA(const GPUSettingsQA* config = nullptr);
  ~GPUO2InterfaceQA();

  // Input might be modified, so we assume non-const. If it is const, a copy should be created before.
  int postprocess(std::vector<TH1F>& in1, std::vector<TH2F>& in2, std::vector<TH1D>& in3, TObjArray& out);

 private:
  std::unique_ptr<GPUQA> mQA;
};
} // namespace o2::gpu

#endif
