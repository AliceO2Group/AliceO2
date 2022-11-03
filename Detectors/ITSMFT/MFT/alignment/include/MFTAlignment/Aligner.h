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

/// \file Aligner.h
/// \author arakotoz@cern.ch
/// \brief Abstract base class for the standalone alignment of MFT

#ifndef ALICEO2_MFT_ALIGNER_H
#define ALICEO2_MFT_ALIGNER_H

#include <array>
#include <vector>

#include <Rtypes.h>
#include <TString.h>
#include <TFile.h>

#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include "MFTAlignment/MillePede2.h"

namespace o2
{
namespace mft
{

class Aligner
{
 public:
  /// \brief construtor
  Aligner();

  /// \brief destructor
  virtual ~Aligner();

  /// \brief init Millipede (will be overriden in derived classes)
  virtual void init() = 0;

  // simple setters

  void setChi2CutNStdDev(const Int_t value) { mChi2CutNStdDev = value; }
  void setResidualCutInitial(const Double_t value) { mResCutInitial = value; }
  void setResidualCut(const Double_t value) { mResCut = value; }
  void setAllowedVariationDeltaX(const double value) { mAllowVar[0] = value; }
  void setAllowedVariationDeltaY(const double value) { mAllowVar[1] = value; }
  void setAllowedVariationDeltaZ(const double value) { mAllowVar[3] = value; }
  void setAllowedVariationDeltaRz(const double value) { mAllowVar[2] = value; }
  void setChi2CutFactor(const double value) { mStartFac = value; }

 protected:
  static constexpr int mNumberOfTrackParam = 4;                                  ///< Number of track (= local) parameters (X0, Tx, Y0, Ty)
  static constexpr int mNDofPerSensor = 4;                                       ///< translation in global x, y, z, and rotation Rz around global z-axis
  static o2::itsmft::ChipMappingMFT mChipMapping;                                ///< MFT chip <-> ladder, layer, disk, half mapping
  static constexpr int mNumberOfSensors = mChipMapping.getNChips();              ///< Total number of sensors (detection elements) in the MFT
  static constexpr int mNumberOfGlobalParam = mNDofPerSensor * mNumberOfSensors; ///< Number of alignment (= global) parameters
  std::array<double, mNDofPerSensor> mAllowVar;                                  ///< "Encouraged" variation for degrees of freedom {dx, dy, dRz, dz}
  double mStartFac;                                                              ///< Initial value for chi2 cut, used to reject outliers i.e. bad tracks with sum(chi2) > Chi2DoFLim(fNStdDev, nDoF) * chi2CutFactor (if > 1, iterations in Millepede are turned on)
  int mChi2CutNStdDev;                                                           ///< Number of standard deviations for chi2 cut
  double mResCutInitial;                                                         ///< Cut on residual on first iteration
  double mResCut;                                                                ///< Cut on residual for other iterations
  TString mMilleRecordsFileName;                                                 ///< output file name when saving the Mille records
  TString mMilleConstraintsRecFileName;                                          ///< output file name when saving the records of the constraints
  bool mIsInitDone = false;                                                      ///< boolean to follow the initialisation status
  std::vector<int> mGlobalParameterStatus;                                       ///< vector of effective degrees of freedom, used to fix detectors, parameters, etc.

  // used to fix some degrees of freedom

  static constexpr int mFixedParId = -1;
  static constexpr int mFreeParId = mFixedParId - 1;

  ClassDef(Aligner, 0);
};

} // namespace mft
} // namespace o2

#endif
