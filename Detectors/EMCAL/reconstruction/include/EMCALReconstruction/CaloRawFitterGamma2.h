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

class CaloRawFitterGamma2 : public CaloRawFitter
{

 public:
  /// \brief Constructor
  CaloRawFitterGamma2();

  /// \brief Destructor
  ~CaloRawFitterGamma2() = default;

  void setNiterationsMax(int n) { mNiterationsMax = n; }
  int getNiterations() { return mNiter; }
  int getNiterationsMax() { return mNiterationsMax; }

  /// \brief Evaluation Amplitude and TOF
  /// return Container with the fit results (amp, time, chi2, ...)
  virtual CaloFitResults evaluate(const std::vector<Bunch>& bunchvector,
                                  std::optional<unsigned int> altrocfg1,
                                  std::optional<unsigned int> altrocfg2);

 private:
  int mNiter = 0;           ///< number of iteraions
  int mNiterationsMax = 15; ///< max number of iteraions

  /// \brief Fits the raw signal time distribution
  /// \return chi2, fit status.
  std::tuple<float, bool> doFit_1peak(int firstTimeBin, int nSamples, float& ampl, float& time);

  /// \brief Fits the raw signal time distribution
  /// \return the fit parameters: amplitude, time.
  std::tuple<float, float> doParabolaFit(int x) const;

  ClassDefNV(CaloRawFitterGamma2, 1);
}; // End of CaloRawFitterGamma2

} // namespace emcal

} // namespace o2
#endif
