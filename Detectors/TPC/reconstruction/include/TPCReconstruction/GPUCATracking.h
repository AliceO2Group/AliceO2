// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCATracking.h
/// \brief Wrapper class for TPC CA Tracker algorithm
/// \author David Rohr
#ifndef ALICEO2_TPC_TPCCATRACKING_H_
#define ALICEO2_TPC_TPCCATRACKING_H_
#include <memory>
#include <vector>
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/TrackTPC.h"

namespace o2
{
namespace gpu
{
struct GPUO2InterfaceConfiguration;
struct GPUO2InterfaceIOPtrs;
class GPUTPCO2Interface;
} // namespace gpu
} // namespace o2

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <class T>
class MCTruthContainer;
}
} // namespace o2

namespace o2
{
namespace tpc
{

class GPUCATracking
{
 public:
  GPUCATracking();
  ~GPUCATracking();
  GPUCATracking(const GPUCATracking&) = delete;            // Disable copy
  GPUCATracking& operator=(const GPUCATracking&) = delete; // Disable assignment

  int initialize(const o2::gpu::GPUO2InterfaceConfiguration& config);
  void deinitialize();

  //Input: cluster structure, possibly including MC labels, pointers to std::vectors for tracks and track MC labels. outputTracksMCTruth may be nullptr to indicate missing cluster MC labels. Otherwise, cluster MC labels are assumed to be present.
  int runTracking(o2::gpu::GPUO2InterfaceIOPtrs* data);

  float getPseudoVDrift();                                              //Return artificial VDrift used to convert time to Z
  int getNTracksASide() { return mNTracksASide; }
  void GetClusterErrors2(int row, float z, float sinPhi, float DzDs, short clusterState, float& ErrY2, float& ErrZ2) const;

 private:
  std::unique_ptr<o2::gpu::GPUTPCO2Interface> mTrackingCAO2Interface; //Pointer to Interface class in HLT O2 CA Tracking library.
                                                                      //The tracking code itself is not included in the O2 package, but contained in the CA library.
                                                                      //The GPUCATracking class interfaces this library via this pointer to GPUTPCO2Interface class.

  static constexpr float sTrackMCMaxFake = 0.1;
  int mNTracksASide = 0;
};

} // namespace tpc
} // namespace o2
#endif
