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

/// \file TRUElectronics.h
/// \brief EMCAL TRUElectronics for the LZEROElectronics
#ifndef ALICEO2_EMCAL_PATCH_H
#define ALICEO2_EMCAL_PATCH_H

#include <array>
#include <vector>
#include <gsl/span>
#include "Rtypes.h"
// #include "DataFormatsEMCAL/Cluster.h"
// #include "DataFormatsEMCAL/Digit.h"
// #include "DataFormatsEMCAL/Cell.h"
// #include "EMCALBase/Geometry.h"

// #include <fairlogger/Logger.h> // for LOG

namespace o2
{

namespace emcal
{

/// \param mADCvalues vector of the ADC values
/// \param previousTimebinADCvalue previous ADC value, popped out (i-4) if i refers to the current Timebin
struct FastOrStruct {
  std::vector<double> mADCvalues;
  double mPreviousTimebinADCvalue = 0.;

  /// Compute current timesum
  double timesum()
  {
    double timesumvalue = 0;
    for (auto ADCvalue : mADCvalues) {
      timesumvalue += ADCvalue;
    }
    return timesumvalue;
  }

  void init()
  {
  }

  // Update internal ADC values (4 timebins)
  void updateADC(double ADCvalue)
  {

    if (mADCvalues.size() == 4) {
      mPreviousTimebinADCvalue = mADCvalues.front();
      mADCvalues.erase(mADCvalues.begin());
    }
    mADCvalues.push_back(ADCvalue);
  }
};

/// \struct TRUElectronics
/// \brief TRUElectronics creator, based on the TRUElectronics
/// \ingroup EMCALsimulation
/// \author Markus Fasel, ORNL
/// \author Simone Ragoni, Creighton U.
/// \date 03/12/2022
///

struct TRUElectronics {

  /// \brief Main constructor
  /// \param patchSize patch size: 2x2, or 4x4
  /// \param whichSide 0 = A side, 1 = C side
  /// \param whichSuperModuleSize 0 = Full 1 = 1/3
  TRUElectronics(int patchSize, int whichSide, int whichSuperModuleSize);

  /// \brief Default constructor
  TRUElectronics();

  /// \brief Destructor
  ~TRUElectronics() = default;

  /// \brief Clear internal members
  void clear();

  /// \brief Initialise internal members
  void init();

  /// \brief Assign seed module to a Full SM
  /// \param patchID Patch ID that need to be assigned a seed
  void assignSeedModuleToPatchWithSTUIndexingFullModule(int& patchID);

  /// \brief Assign seed module to a 1/3 SM
  /// \param patchID Patch ID that need to be assigned a seed
  void assignSeedModuleToPatchWithSTUIndexingOneThirdModule(int& patchID);

  /// \brief Assign seed module to all patches
  void assignSeedModuleToAllPatches();

  /// \brief Assign modules to all patches
  void assignModulesToAllPatches();

  /// \brief Updates the patches
  void updateADC();

  int mPatchSize;            //!<! patch size (2x2 or 4x4 typically)
  int mWhichSide;            //!<! Either A = 0 or C = 1 side
  int mWhichSuperModuleSize; //!<! Either Full/2/3 = 0 or 1/3 = 1 size

  std::vector<std::tuple<int, int>> mPatchIDSeedFastOrIDs;                  //!<! mask containing Patch IDs, their seed FastOrs
  std::vector<std::tuple<int, std::vector<int>>> mIndexMapPatch;            //!<! mask of the  FastOrs assigned to each patch
  std::vector<std::tuple<int, std::vector<int>>> mFiredFastOrIndexMapPatch; //!<! mask of the FastOrs above threshold in each patch
  std::vector<int> mFiredPatches;                                           //!<! mask of the patches above threshold
  std::vector<std::tuple<int, std::vector<double>>> mADCvalues;             //!<! ADC values for peak finding
  std::vector<std::tuple<int, std::vector<double>>> mTimesum;               //!<! Time sums for peak finding
  std::vector<std::tuple<int, double>> mPreviousTimebinADCvalue;            //!<! ADC that was just removed from the time bins
  std::vector<FastOrStruct> mFastOrs;                                       //!<! FastOr objects

  ClassDefNV(TRUElectronics, 1);
};

} // namespace emcal
} // namespace o2
#endif /* ALICEO2_EMCAL_PATCH_H */
