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

///
/// @file   PID.h
/// @author Thomas Klemenz, thomas.klemenz@tum.de
///

#ifndef AliceO2_TPC_QC_PID_H
#define AliceO2_TPC_QC_PID_H

#include <vector>
#include <unordered_map>
#include <memory>
#include <string_view>

// root includes
#include "TH1.h"
#include "TCanvas.h"

// o2 includes
#include "DataFormatsTPC/Defs.h"

namespace o2
{
namespace tpc
{

class TrackTPC;

namespace qc
{

/// @brief  PID quality control class
///
/// This class is used to extract PID related variables
/// from TrackTPC objects and store it in histograms.
///
/// origin: TPC
/// @author Thomas Klemenz, thomas.klemenz@tum.de
class PID
{
 public:
  /// \brief Constructor.
  PID() = default;

  /// bool extracts intormation from track and fills it to histograms
  /// @return true if information can be extracted and filled to histograms
  bool processTrack(const o2::tpc::TrackTPC& track, size_t NTracks);

  /// Initialize all histograms
  void initializeHistograms();

  /// Reset all histograms
  void resetHistograms();

  /// Dump results to a file
  void dumpToFile(std::string filename);

  // To set the elementary track cuts
  void setPIDCuts(int minnCls = 60, float absTgl = 1., float mindEdxTot = 10.0,
                  float maxdEdxTot = 70., float minpTPC = 0.05, float maxpTPC = 20., float minpTPCMIPs = 0.45, float maxpTPCMIPs = 0.55)
  {
    mCutMinnCls = minnCls;
    mCutAbsTgl = absTgl;
    mCutMindEdxTot = mindEdxTot;
    mCutMaxdEdxTot = maxdEdxTot;
    mCutMinpTPC = minpTPC;
    mCutMaxpTPC = maxpTPC;
    mCutMinpTPCMIPs = minpTPCMIPs;
    mCutMaxpTPCMIPs = maxpTPCMIPs;
  }
  std::unordered_map<std::string_view, std::vector<std::unique_ptr<TH1>>>& getMapOfHisto() { return mMapHist; }
  std::unordered_map<std::string_view, std::vector<std::unique_ptr<TCanvas>>>& getMapOfCanvas() { return mMapCanvas; }
  const std::unordered_map<std::string_view, std::vector<std::unique_ptr<TH1>>>& getMapOfHisto() const { return mMapHist; }
  const std::unordered_map<std::string_view, std::vector<std::unique_ptr<TCanvas>>>& getMapOfCanvas() const { return mMapCanvas; }

 private:
  int mCutMinnCls = 60;          // minimum N clusters
  float mCutAbsTgl = 1.f;        // AbsTgl max cut
  float mCutMindEdxTot = 10.f;   // dEdxTot min value
  float mCutMaxdEdxTot = 70.f;   // dEdxTot max value
  float mCutMinpTPC = 0.05f;     // pTPC min value
  float mCutMaxpTPC = 20.f;      // pTPC max value
  float mCutMinpTPCMIPs = 0.45f; // pTPC min value for MIPs
  float mCutMaxpTPCMIPs = 0.55f; // pTPC max value for MIPs
  // ===| The following parameters are used to set up the getMostProbablePID() function. It needs some hardcoded cuts in the overlap region.
  // ===| They are taken from O2/GPU/GPUTracking/Definitions/GPUSettingsList.h
  float mCutKrangeMin = 0.47f; // minMomentum for Kaons in dEdx overlap region
  float mCutKrangeMax = 0.57f; // maxMomentum for Kaons in dEdx overlap region
  float mCutPrangeMin = 0.93f; // minMomentum for Protons in dEdx overlap region
  float mCutPrangeMax = 1.03f; // maxMomentum for Protons in dEdx overlap region
  float mCutDrangeMin = 1.88f; // minMomentum for Deuterons in dEdx overlap region
  float mCutDrangeMax = 1.98f; // maxMomentum for Deuterons in dEdx overlap region
  float mCutTrangeMin = 2.84f; // minMomentum for Tritons in dEdx overlap region
  float mCutTrangeMax = 2.94f; // maxMomentum for Tritons in dEdx overlap region
  float mPIDSigma = 0.06f;     // relative sigma for PID in combination with UseNSigma (hardcoded to 1)
  std::unordered_map<std::string_view, std::vector<std::unique_ptr<TH1>>> mMapHist;
  // Map for Canvases to be published
  std::unordered_map<std::string_view, std::vector<std::unique_ptr<TCanvas>>> mMapCanvas;
  // Map for Histograms which will be put onto the canvases, and not published separately
  std::unordered_map<std::string_view, std::vector<std::unique_ptr<TH1>>> mMapHistCanvas;
  ClassDefNV(PID, 1)
};
} // namespace qc
} // namespace tpc
} // namespace o2

#endif