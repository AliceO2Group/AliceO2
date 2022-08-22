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
  ~RecoParam() override = default;

  double getCellTimeShiftNanoSec() const { return mCellTimeShiftNanoSec; }
  double getNoiseThresholdLGnoHG() const { return mNoiseThresholdLGnoHG; }
  int getPhaseBCmod4() const { return mPhaseBCmod4; }

  void PrintStream(std::ostream& stream) const;

 private:
  double mNoiseThresholdLGnoHG = 10.;  ///< Noise threshold applied to suppress LGnoHG error
  double mCellTimeShiftNanoSec = 470.; ///< Time shift applied on the cell time to center trigger peak around 0
  int mPhaseBCmod4 = 1;                ///< Rollback of the BC ID in the correction of the cell time for the BC mod 4

  O2ParamDef(RecoParam, "EMCRecoParam");
};
std::ostream& operator<<(std::ostream& stream, const RecoParam& s);
} // namespace emcal
} // namespace o2
#endif