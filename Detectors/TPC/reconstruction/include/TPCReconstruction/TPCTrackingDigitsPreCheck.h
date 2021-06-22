// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCTrackingDigitsPreCheck.h
/// \brief Wrapper class for TPC CA Tracker algorithm
/// \author David Rohr

#ifndef ALICEO2_TPC_TRACKING_DIGITSPRECHECK_H
#define ALICEO2_TPC_TRACKING_DIGITSPRECHECK_H

#include <memory>
namespace o2
{
namespace gpu
{
struct GPUTrackingInOutPointers;
struct GPUO2InterfaceConfiguration;
} // namespace gpu
} // namespace o2

namespace o2
{
namespace tpc
{

class TPCTrackingDigitsPreCheck
{
  struct precheckModifiedDataInternal; // This struct might hold some internal data which the modified members of the data argument might point to

 public:
  class precheckModifiedData
  {
    std::unique_ptr<precheckModifiedDataInternal> data;

   public:
    precheckModifiedData();
    precheckModifiedData(std::unique_ptr<precheckModifiedDataInternal>&& v);
    ~precheckModifiedData();
  };
  static precheckModifiedData runPrecheck(o2::gpu::GPUTrackingInOutPointers* ptrs, o2::gpu::GPUO2InterfaceConfiguration* config);
};

} // namespace tpc
} // namespace o2
#endif // ALICEO2_TPC_TRACKING_DIGITSPRECHECK_H
