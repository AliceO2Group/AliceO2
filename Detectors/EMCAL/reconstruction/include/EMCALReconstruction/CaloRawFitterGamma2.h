// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef __CALORAWFITTERGAMMA2_H__
#define __CALORAWFITTERGAMMA2_H__

#include <iosfwd>
#include <array>
#include <optional>
#include <Rtypes.h>
#include "EMCALReconstruction/CaloFitResults.h"
#include "DataFormatsEMCAL/Constants.h"
#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloRawFitter.h"

namespace o2
{

namespace emcal
{

/// \class CaloRawFitterGamma2
/// \brief  Raw data fitting: Gamma-2 function
/// \ingroup EMCALreconstruction
/// \author Martin Poghosyan <Martin.Poghosyan@cern.ch>, ORNL.
/// \since May 13th, 2020
///
/// Evaluation of amplitude and peak position using gamma-2 function.
/// Derivatives calculated analytically.
/// Newton's method used for solving the set of non-linear equations.
/// Ported from class AliCaloRawAnalyzerGamma2 from AliRoot

class CaloRawFitterGamma2 final : public CaloRawFitter
{

 public:
  /// \brief Constructor
  CaloRawFitterGamma2();

  /// \brief Destructor
  ~CaloRawFitterGamma2() final = default;

  void setNiterationsMax(int n) { mNiterationsMax = n; }
  int getNiterations() { return mNiter; }
  int getNiterationsMax() { return mNiterationsMax; }

  /// \brief Evaluation Amplitude and TOF
  /// \param bunchvector ALTRO bunches for the current channel
  /// \param altrocfg1 ALTRO config register 1 from RCU trailer
  /// \param altrocfg2 ALTRO config register 2 from RCU trailer
  /// \throw RawFitterError_t::FIT_ERROR in case the peak fit failed
  /// \return Container with the fit results (amp, time, chi2, ...)
  CaloFitResults evaluate(const gsl::span<const Bunch> bunchvector,
                          std::optional<unsigned int> altrocfg1,
                          std::optional<unsigned int> altrocfg2) final;

 private:
  int mNiter = 0;           ///< number of iteraions
  int mNiterationsMax = 15; ///< max number of iteraions

  /// \brief Fits the raw signal time distribution
  /// \param firstTimeBin First timebin in the ALTRO bunch
  /// \param nSamples Number of time samples of the ALTRO bunch
  /// \param[in] ampl Initial guess of the amplitude for the fit
  /// \param[out] ampl Amplitude result of the peak fit
  /// \param[in] time Initial guess of the time for the fit
  /// \param[out] time Time result of the peak fit
  /// \return chi2 of the fit
  /// \throw RawFitterError_t::FIT_ERROR in case of fit errors (insufficient number of time samples, matrix diagonalization error, ...)
  float doFit_1peak(int firstTimeBin, int nSamples, float& ampl, float& time);

  /// \brief Fits the raw signal time distribution
  /// \param maxTimeBin Time bin of the max. amplitude
  /// \return the fit parameters: amplitude, time.
  ///
  /// Fit performed as parabola fit to the signal
  std::tuple<float, float> doParabolaFit(int maxTimeBin) const;

  ClassDefNV(CaloRawFitterGamma2, 1);
}; // End of CaloRawFitterGamma2

} // namespace emcal

} // namespace o2
#endif
