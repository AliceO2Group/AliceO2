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

/// \class CaloRawFitterGSGS
/// \brief  Raw data fitting based on NIM A621 (2010) 231–237
///
/// Extraction of amplitude and time
/// from CALO raw data using analytical calculation of
/// least square fit with Gamma2 function
///
/// \author Dmitri Peresunko after M.Bogolybski
/// \since April.2021
///

#ifndef PHOSRAWFITTERGS_H
#define PHOSRAWFITTERGS_H
#include "PHOSReconstruction/CaloRawFitter.h"

namespace o2
{

namespace phos
{

class CaloRawFitterGS : public CaloRawFitter
{

 public:
  static constexpr int NMAXSAMPLES = 40; ///< maximal expected number of samples per bunch
  /// \brief Constructor
  CaloRawFitterGS();

  /// \brief Destructor
  ~CaloRawFitterGS() final = default;

  /// \brief Evaluation Amplitude and TOF
  FitStatus evaluate(gsl::span<short unsigned int> signal) final;

 protected:
  void init();
  FitStatus evalFit(gsl::span<short unsigned int> signal);

 private:
  short mMinTimeCalc = 10;      ///< minimal sample amplitude to calculate time and amp
  float mDecTime = 0.058823529; ///< decay time constant
  float mQAccuracy = 1.e-3;     ///< accuracy of iterative fit
  float mexp[NMAXSAMPLES];      ///< arrays to tabulate Gamma2 function and its momenta
  float ma0[NMAXSAMPLES];       ///< arrays to tabulate Gamma2 function and its momenta
  float ma1[NMAXSAMPLES];       ///< arrays to tabulate Gamma2 function and its momenta
  float ma2[NMAXSAMPLES];       ///< arrays to tabulate Gamma2 function and its momenta
  float ma3[NMAXSAMPLES];       ///< arrays to tabulate Gamma2 function and its momenta
  float ma4[NMAXSAMPLES];       ///< arrays to tabulate Gamma2 function and its momenta

  ClassDef(CaloRawFitterGS, 2);
}; // End of CaloRawFitterGS

} // namespace phos

} // namespace o2
#endif
