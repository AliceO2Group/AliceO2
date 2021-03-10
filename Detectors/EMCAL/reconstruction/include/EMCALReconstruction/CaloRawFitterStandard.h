// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef EMCALRAWFITTERSTANDARD_H_
#define EMCALRAWFITTERSTANDARD_H_

#include <iosfwd>
#include <array>
#include <optional>
#include <Rtypes.h>
#include "EMCALReconstruction/CaloFitResults.h"
#include "DataFormatsEMCAL/Constants.h"
#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloRawFitter.h"

class TGraph;

namespace o2
{

namespace emcal
{

/// \class CaloRawFitterStandard
/// \brief  Raw data fitting: standard TMinuit fit
/// \ingroup EMCALreconstruction
/// \author Hadi Hassan <hadi.hassan@cern.ch>, Oak Ridge National Laboratory
/// \since November 4th, 2019
///
/// Extraction of amplitude and peak position
/// from CALO raw data using
/// least square fit for the
/// Moment assuming identical and
/// independent errors (equivalent with chi square)
class CaloRawFitterStandard final : public CaloRawFitter
{

 public:
  /// \brief Constructor
  CaloRawFitterStandard();

  /// \brief Destructor
  ~CaloRawFitterStandard() final = default;

  /// \brief Approximate response function of the EMCal electronics.
  /// \param x bin
  /// \param par function parameters
  /// \return double with signal for a given time bin
  static double rawResponseFunction(double* x, double* par);

  /// \brief Evaluation Amplitude and TOF
  /// \param bunchvector Calo bunches for the tower and event
  /// \param altrocfg1 ALTRO config register 1 from RCU trailer
  /// \param altrocfg2 ALTRO config register 2 from RCU trailer
  /// \return Container with the fit results (amp, time, chi2, ...)
  /// \throw RawFitterError_t in case the fit failed (including all possible errors from upstream)
  CaloFitResults evaluate(const gsl::span<const Bunch> bunchvector,
                          std::optional<unsigned int> altrocfg1,
                          std::optional<unsigned int> altrocfg2) final;

  /// \brief Fits the raw signal time distribution using TMinuit
  /// \param firstTimeBin First timebin of the ALTRO bunch
  /// \param lastTimeBin Last timebin of the ALTRO bunch
  /// \return the fit parameters: amplitude, time, chi2
  /// \throw RawFitter_t::FIT_ERROR in case the fit failed (insufficient number of samples or fit error from MINUIT)
  std::tuple<float, float, float> fitRaw(int firstTimeBin, int lastTimeBin) const;

 private:
  ClassDefNV(CaloRawFitterStandard, 1);
}; // End of CaloRawFitterStandard

} // namespace emcal

} // namespace o2
#endif
