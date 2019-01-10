// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCCATracking.h
/// \brief Wrapper class for TPC CA Tracker algorithm
/// \author David Rohr
#ifndef ALICEO2_TPC_TPCCATRACKING_H_
#define ALICEO2_TPC_TPCCATRACKING_H_
#include <memory>
#include <vector>
#include "DataFormatsTPC/Cluster.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/TrackTPC.h"

class AliGPUCAConfiguration;
class AliHLTTPCCAO2Interface;

namespace o2 { class MCCompLabel; namespace dataformats { template <class T> class MCTruthContainer; }}

namespace o2
{
namespace TPC
{

class TPCCATracking
{
public:
  TPCCATracking();
  ~TPCCATracking();

  int initialize(const AliGPUCAConfiguration& config);
  int initialize(const char* options = nullptr);
  void deinitialize();

  //Input: cluster structure, possibly including MC labels, pointers to std::vectors for tracks and track MC labels. outputTracksMCTruth may be nullptr to indicate missing cluster MC labels. Otherwise, cluster MC labels are assumed to be present.
  int runTracking(const o2::TPC::ClusterNativeAccessFullTPC& clusters, std::vector<TrackTPC>* outputTracks,
                  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* outputTracksMCTruth = nullptr);

  float getPseudoVDrift();                                            //Return artificial VDrift used to convert time to Z
  float getTFReferenceLength() {return sContinuousTFReferenceLength;} //Return reference time frame length used to obtain Z from T in continuous data
  int getNTracksASide() {return mNTracksASide;}
  void GetClusterErrors2(int row, float z, float sinPhi, float DzDs, float& ErrY2, float& ErrZ2) const;

 private:
  std::unique_ptr<AliHLTTPCCAO2Interface> mTrackingCAO2Interface; //Pointer to Interface class in HLT O2 CA Tracking library.
                                                                  //The tracking code itself is not included in the O2 package, but contained in the CA library.
                                                                  //The TPCCATracking class interfaces this library via this pointer to AliHLTTPCCAO2Interface class.

  TPCCATracking(const TPCCATracking&) = delete;            // Disable copy
  TPCCATracking& operator=(const TPCCATracking&) = delete; // Disable assignment
  
  static constexpr float sContinuousTFReferenceLength = 0.023 * 5e6;
  static constexpr float sTrackMCMaxFake = 0.1;
  int mNTracksASide = 0;
};

}
}
#endif
