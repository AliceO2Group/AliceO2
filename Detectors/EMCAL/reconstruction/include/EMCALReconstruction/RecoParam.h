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
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_RECOPARAM_H_
#define ALICEO2_EMCAL_RECOPARAM_H_

#include <iosfwd>
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "Rtypes.h"

namespace o2
{
namespace emcal
{
/// \class RecoParam
/// \brief EMCal reconstruction parameters
/// \ingroup EMCALreconstruction
class RecoParam : public o2::conf::ConfigurableParamHelper<RecoParam>
{
 public:
  /// \brief Destructor
  ~RecoParam() override = default;

  /// \brief Get the average cell time shift
  /// \return Average cell time shift
  double getCellTimeShiftNanoSec() const { return mCellTimeShiftNanoSec; }

  /// \brief Get noise threshold for LGnoHG error
  /// \return Noise threshold
  double getNoiseThresholdLGnoHG() const { return mNoiseThresholdLGnoHG; }

  /// \brief Get the BC phase
  /// \return BC phase
  int getPhaseBCmod4() const { return mPhaseBCmod4; }

  /// \brief Get the max. allowed bunch length
  /// \return Max. allowed bunch length
  ///
  /// In case of max. bunch length 0 the max. bunch length is auto-determined from
  /// the RCU trailer
  int getMaxAllowedBunchLength() const { return mMaxBunchLength; }

  /// \brief Print current reconstruction parameters to stream
  /// \param stream Stream to print on
  void PrintStream(std::ostream& stream) const;

 private:
  double mNoiseThresholdLGnoHG = 10.;  ///< Noise threshold applied to suppress LGnoHG error
  double mCellTimeShiftNanoSec = 470.; ///< Time shift applied on the cell time to center trigger peak around 0
  int mPhaseBCmod4 = 1;                ///< Rollback of the BC ID in the correction of the cell time for the BC mod 4
  unsigned int mMaxBunchLength = 15;   ///< Max. allowed bunch length

  O2ParamDef(RecoParam, "EMCRecoParam");
};

/// \brief Streaming operator for the reconstruction parameters
/// \param stream Stream to print on
/// \param par RecoParams to be printed
/// \return Stream after printing the reco params
std::ostream& operator<<(std::ostream& stream, const RecoParam& par);
} // namespace emcal

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::emcal::RecoParam> : std::true_type {
};
} // namespace framework

} // namespace o2
#endif