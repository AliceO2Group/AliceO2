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
/// \file   PIDTPC.h
/// \author Nicolo' Jacazio
///

#ifndef O2_FRAMEWORK_PIDTPC_H_
#define O2_FRAMEWORK_PIDTPC_H_

// ROOT includes
#include "Rtypes.h"
#include "TMath.h"

// O2 includes
#include "Framework/Logger.h"
#include "ReconstructionDataFormats/PID.h"
#include "PID/ParamBase.h"

namespace o2::pid::tpc
{

float BetheBlochF(float x, const float p[5]);
float RelResolutionF(float x, const float p[2]);

/// \brief Class to handle the parametrization of the detector response
class Param
{
 public:
  Param() = default;
  ~Param() = default;

  /// Getter for the expected signal
  /// Calculates the expected PID signal as the function of
  /// the information stored in the track and the given parameters,
  /// for the specified particle type
  ///
  /// At the moment, these signals are just the results of calling the
  /// Bethe-Bloch formula plus, if desired, taking into account the eta dependence
  /// and the multiplicity dependence (for PbPb).
  /// This can be improved. By taking into account the number of
  /// assigned clusters and/or the track dip angle, for example.
  float GetExpectedSignal(float mom, float mass, float charge) const;

  /// Getter for the charge factor
  /// BB goes with z^2, however in reality it is slightly larger (calibration, threshold effects, ...)
  /// !!! Splines for light nuclei need to be normalised to this factor !!!
  float GetChargeFactor(float charge) const { return TMath::Power(charge, 2.3); }

  /// Getter for the expected resolution.
  /// Returns the expected sigma of the PID signal.
  /// If the operation is not possible, return a negative value.
  float GetExpectedSigma(float npoints, float tpcsignal) const;

  /// This is the Bethe-Bloch function normalised to 1 at the minimum
  /// WARNING
  /// Simulated and reconstructed Bethe-Bloch differs
  ///           Simulated  curve is the dNprim/dx
  ///           Reconstructed is proportianal dNtot/dx
  /// Temporary fix for production -  Simple linear correction function
  /// Future    2 Bethe Bloch formulas needed
  ///           1. for simulation
  ///           2. for reconstructed PID
  Parametrization<float, 5, BetheBlochF> mBetheBloch = Parametrization<float, 5, BetheBlochF>();

  Parametrization<float, 2, RelResolutionF> mRelResolution = Parametrization<float, 2, RelResolutionF>();

  float fMIP = 50.f; // dEdx for MIP
 private:
  //   float fRes0[fgkNumberOfGainScenarios];  // relative dEdx resolution  rel sigma = fRes0*sqrt(1+fResN2/npoint)
  //   float fResN2[fgkNumberOfGainScenarios]; // relative Npoint dependence rel  sigma = fRes0*sqrt(1+fResN2/npoint)
};

/// \brief Class to handle the the TPC detector response
class Response
{
 public:
  Response() = default;
  ~Response() = default;

  /// Updater for the TPC response to setup the track parameters
  /// i.e. sets the track of interest
  void UpdateTrack(float mom, float tpcsignal, float tpcpoints)
  {
    mMomentum = mom;
    mTPCSignal = tpcsignal;
    mTPCPoints = tpcpoints;
  };

  // Expected resolution
  /// Gets the expected resolution of the measurement
  float GetExpectedSigma(o2::track::PID::ID id) const { return mParam.GetExpectedSigma(mTPCSignal, mTPCPoints); }

  // Expected signal
  /// Gets the expected signal of the measurement
  float GetExpectedSignal(o2::track::PID::ID id) const { return mParam.GetExpectedSignal(mMomentum, o2::track::PID::getMass(id), o2::track::PID::getCharge(id)); }

  // Nsigma
  float GetNumberOfSigmas(o2::track::PID::ID id) const { return (mTPCSignal - GetExpectedSignal(id)) / GetExpectedSigma(id); }

  Param mParam; /// Parametrization of the TPC signal
 private:
  // Event of interest information
  // Track of interest information
  float mMomentum;  /// Momentum
  float mTPCSignal; /// TPC signal
  float mTPCPoints; /// Number of TPC points for TPC signal
};

} // namespace o2::pid::tpc

#endif // O2_FRAMEWORK_PIDTPC_H_
