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

/// \file ClusterConfig.h
/// \brief Clustering and fifting parameters
/// \author Gilles Grasseau, Subatech

#ifndef O2_MCH_CLUSTERCONFIG_H_
#define O2_MCH_CLUSTERCONFIG_H_

namespace o2
{
namespace mch
{

typedef int PadIdx_t;   // Pad index type
typedef short Groups_t; // Groups/sub-cluster type
typedef short Mask_t;   // To build mask

struct ClusterConfig {
  //
  // Physical-Numerical parameters
  //
  // Run2
  // 4.f * 0.22875f;
  double minChargeOfPads;              // Lowest Charge of a Pad
  double minChargeOfClusterPerCathode; // Lowest Charge of a Pad
  // static constexpr double minChargeOfClusterPerCathode = 1.1; // Lowest Charge of a Group
  // Run3
  // static double minChargeOfPads = 16; // Lowest Charge of a Pad
  // static double minChargeOfClusterPerCathode = 1.0 * minChargeOfPads; // Lowest Charge of a Group
  //
  // ClusterResolution
  float SDefaultClusterResolution; ///< default cluster resolution (cm)
  float SBadClusterResolution;     ///< bad (e.g. mono-cathode) cluster resolution (cm)

  // Large Clusters
  int nbrPadLimit = 600;
  double ratioStepForLargeCluster = 0.05; // increment to find nPads < nbrPadLimit
  // Limit of pad number  to perform the fitting
  int nbrOfPadsLimitForTheFitting = 100;
  // Stop the fitting if small xy shift
  double minFittingXYStep = 0.1; // in cm
  //
  // Algorithm choices
  //
  int useSpline = 0;
  // Logs
  //
  enum VerboseMode {
    no = 0x0,     ///< No message
    info = 0x1,   ///< Describes main steps and high level behaviors
    detail = 0x2, ///< Describes in detail
    debug = 0x3   ///< Ful details
  };
  VerboseMode fittingLog = no;
  VerboseMode processingLog = info; // Global
  VerboseMode padMappingLog = no;
  VerboseMode groupsLog = no;
  VerboseMode EMLocalMaxLog = no;
  VerboseMode inspectModelLog = no;
  VerboseMode laplacianLocalMaxLog = no;
  //
  // Checks
  //
  enum ActivateMode {
    inactive = 0x0, ///< No activation
    active = 0x1,   ///< Describe default activation
  };
  // Activate/deactivate InspectModel
  ActivateMode inspectModel = inactive;
  //
  bool groupsCheck = true;
  bool padMappingCheck = true;
  bool mathiesonCheck = false;
};

void initClusterConfig();

} // namespace mch
} // end namespace o2

#endif // O2_MCH_CLUSTERCONFIG_H_
