// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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
class TClonesArray;
class AliHLTTPCCAO2Interface;
class AliHLTTPCCAClusterData;

namespace o2
{
namespace TPC
{

class TrackTPC;

class TPCCATracking
{
public:
  TPCCATracking();
  ~TPCCATracking();

  int initialize(const char* options = nullptr);
  void deinitialize();

  int runTracking(const TClonesArray* inputClusters, std::vector<TrackTPC>* outputTracks);

private:
  std::unique_ptr<AliHLTTPCCAO2Interface> mTrackingCAO2Interface; //Pointer to Interface class in HLT O2 CA Tracking library.
                                                                  //The tracking code itself is not included in the O2 package, but contained in the CA library.
                                                                  //The TPCCATracking class interfaces this library via this pointer to AliHLTTPCCAO2Interface class.
  std::unique_ptr<AliHLTTPCCAClusterData[]> mClusterData_UPTR;
  AliHLTTPCCAClusterData* mClusterData;

  TPCCATracking(const TPCCATracking&) = delete;            // Disable copy
  TPCCATracking& operator=(const TPCCATracking&) = delete; // Disable assignment
};

}
}
#endif
