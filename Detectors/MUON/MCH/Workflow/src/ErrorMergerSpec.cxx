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

/// \file ErrorMergerSpec.cxx
/// \brief Implementation of a data processor to merge the processing errors in one single output
///
/// \author Philippe Pillot, Subatech

#include "ErrorMergerSpec.h"

#include <vector>

#include <gsl/span>

#include "Framework/ConfigParamSpec.h"
#include "Framework/InitContext.h"
#include "Framework/InputSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/OutputRef.h"
#include "Framework/OutputSpec.h"
#include "Framework/ProcessingContext.h"
#include "Framework/Task.h"

#include "MCHBase/Error.h"

namespace o2
{
namespace mch
{

using namespace o2::framework;

class ErrorMergerTask
{
 public:
  //_________________________________________________________________________________________________
  /// constructor
  ErrorMergerTask(bool preclustering, bool clustering, bool tracking)
    : mPreclustering{preclustering},
      mClustering{clustering},
      mTracking{tracking} {}

  //_________________________________________________________________________________________________
  /// prepare the error merger
  void init(InitContext& ic) {}

  //_________________________________________________________________________________________________
  /// run the error merger
  void run(ProcessingContext& pc)
  {
    auto& errors = pc.outputs().make<std::vector<Error>>(OutputRef{"errors"});

    if (mPreclustering) {
      auto preclusteringErrors = pc.inputs().get<gsl::span<Error>>("preclustererrors");
      errors.insert(errors.end(), preclusteringErrors.begin(), preclusteringErrors.end());
    }

    if (mClustering) {
      auto clusteringErrors = pc.inputs().get<gsl::span<Error>>("clustererrors");
      errors.insert(errors.end(), clusteringErrors.begin(), clusteringErrors.end());
    }

    if (mTracking) {
      auto trackingErrors = pc.inputs().get<gsl::span<Error>>("trackerrors");
      errors.insert(errors.end(), trackingErrors.begin(), trackingErrors.end());
    }
  }

 private:
  bool mPreclustering = true; ///< add preclustering errors
  bool mClustering = true;    ///< add clustering errors
  bool mTracking = true;      ///< add tracking errors
};

//_________________________________________________________________________________________________
DataProcessorSpec getErrorMergerSpec(const char* specName, bool preclustering, bool clustering, bool tracking)
{
  std::vector<InputSpec> inputSpecs{};
  if (preclustering) {
    inputSpecs.emplace_back(InputSpec{"preclustererrors", "MCH", "PRECLUSTERERRORS", 0, Lifetime::Timeframe});
  }
  if (clustering) {
    inputSpecs.emplace_back(InputSpec{"clustererrors", "MCH", "CLUSTERERRORS", 0, Lifetime::Timeframe});
  }
  if (tracking) {
    inputSpecs.emplace_back(InputSpec{"trackerrors", "MCH", "TRACKERRORS", 0, Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    specName,
    inputSpecs,
    Outputs{OutputSpec{{"errors"}, "MCH", "ERRORS", 0, Lifetime::Timeframe}},
    adaptFromTask<ErrorMergerTask>(preclustering, clustering, tracking),
    Options{}};
}

} // namespace mch
} // namespace o2
