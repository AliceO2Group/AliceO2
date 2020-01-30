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
class CaloRawFitterStandard : public CaloRawFitter
{

 public:
  /// \brief Constructor
  CaloRawFitterStandard();

  /// \brief Destructor
  ~CaloRawFitterStandard() = default;

  /// \brief Approximate response function of the EMCal electronics.
  /// \param x: bin
  /// \param par: function parameters
  /// \return double with signal for a given time bin
  static double rawResponseFunction(double* x, double* par);

  /// \brief Evaluation Amplitude and TOF
  /// return Container with the fit results (amp, time, chi2, ...)
  virtual CaloFitResults evaluate(const std::vector<Bunch>& bunchvector,
                                  std::optional<unsigned int> altrocfg1,
                                  std::optional<unsigned int> altrocfg2);

  /// \brief Fits the raw signal time distribution
  /// \return the fit parameters: amplitude, time, chi2, fit status.
  std::tuple<float, float, float, bool> fitRaw(int firstTimeBin, int lastTimeBin) const;

 private:
  ClassDefNV(CaloRawFitterStandard, 1);
}; // End of CaloRawFitterStandard

} // namespace emcal

} // namespace o2
#endif
