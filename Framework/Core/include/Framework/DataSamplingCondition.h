// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DATASAMPLINGCONDITION_H
#define ALICEO2_DATASAMPLINGCONDITION_H

/// \file DataSamplingCondition.h
/// \brief A standarised data sampling condition, to decide if given data sample should be passed forward.
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/DataRef.h"
#include "Configuration/Tree.h"

namespace o2
{
namespace framework
{

/// A standarised data sampling condition, to decide if given data sample should be passed forward.
class DataSamplingCondition
{
 public:
  /// \brief Default constructor.
  DataSamplingCondition() = default;
  /// \brief Default destructor
  virtual ~DataSamplingCondition() = default;

  /// \brief Reads custom configuration parameters.
  virtual void configure(const o2::configuration::tree::Branch&) = 0;
  /// \brief Makes decision whether to pass a data sample or not.
  virtual bool decide(const o2::framework::DataRef&) = 0;

  // a list of getters of specific DataSamplingCondition's
  /// \brief Getter for DataSamplingConditionRandom
  static std::unique_ptr<DataSamplingCondition> getDataSamplingConditionRandom();
};

} // namespace framework
} // namespace o2

#endif //ALICEO2_DATASAMPLINGCONDITION_H
