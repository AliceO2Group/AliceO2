// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   Tracking.h
/// @author David Rohr
///

#ifndef AliceO2_TPC_QC_TRACKING_H
#define AliceO2_TPC_QC_TRACKING_H

#include <vector>
#include <memory>

class TH1F;
class TH2F;
class TH1D;

//o2 includes
#include "DataFormatsTPC/Defs.h"

namespace o2
{
class MCCompLabel;
namespace gpu
{
class GPUO2InterfaceQA;
struct GPUO2InterfaceConfiguration;
} // namespace gpu
namespace tpc
{
class TrackTPC;
struct ClusterNativeAccess;

namespace qc
{
// Class for tracking QA (efficiency / resolution)
// Some settings can be steered via --configKeyValues: (See GPUSettingsList.h for actual definitions). Relevant QA parameters are:
// "GPU_QA.strict=[bool]"               Strict QA mode: Only consider resolution of tracks where the fit ended within 5 cm of the reference, and remove outliers. (Default: true)
// "GPU_QA.qpt=[float]"                 Set cut for Q/Pt. (Default: 10.0)
// "GPU_QA.recThreshold=[float]"        Compute the efficiency including impure tracks with fake contamination. (Default 0.9)
// "GPU_QA.maxResX=[float]"             Maxmimum X (~radius) for reconstructed track position to take into accound for resolution QA in cm (Default: no limit)
// "GPU_QA.nativeFitResolutions=[bool]" Create resolution histograms in the native fit units (sin(phi), tan(lambda), Q/Pt) (Default: false)
// "GPU_QA.filterCharge=[int]"          Filter for positive (+1) or negative (-1) charge (Default: no filter)
// "GPU_QA.filterPID=[int]"             Filter for Particle Type (0 Electron, 1 Muon, 2 Pion, 3 Kaon, 4 Proton) (Default: no filter)

class Tracking
{
 public:
  /// default constructor
  Tracking();
  ~Tracking();

  enum outputModes {
    outputMergeable,     // output mergeaable histogrems, which can be merged and then postprocessed
    outputPostprocessed, // directly postprocess the histograms before merging
    outputLayout         // arrange postprocessed histograms in predefined layouts
  };

  // Initiaalize
  // postprocessOnly = false: initialize to run the full QA via processTracks function.
  // postprocessOnly = true : cannot process tracks but only postprocess mergeeablee histogrems in postprocess function, output type must be outputPostprocessed or outputLayout.
  void initialize(outputModes outputMode, bool postprocessOnly = false);

  void processTracks(const std::vector<o2::tpc::TrackTPC>* tracks, const std::vector<o2::MCCompLabel>* tracksMC, const o2::tpc::ClusterNativeAccess* clNative, TObjArray* out = nullptr);
  int postprocess(std::vector<TH1F>& in1, std::vector<TH2F>& in2, std::vector<TH1D>& in3, TObjArray& out); // Inputs are modified, thus must not be const

  /// Reset all histograms
  void resetHistograms();

  /// get histograms
  void getHists(const std::vector<TH1F>*& h1, const std::vector<TH2F>*& h2, const std::vector<TH1D>*& h3) const;

 private:
  std::unique_ptr<o2::gpu::GPUO2InterfaceConfiguration> mQAConfig; //!
  std::unique_ptr<o2::gpu::GPUO2InterfaceQA> mQA; //!
  outputModes mOutputMode;

  ClassDefNV(Tracking, 1)
};
} // namespace qc
} // namespace tpc
} // namespace o2

#endif
