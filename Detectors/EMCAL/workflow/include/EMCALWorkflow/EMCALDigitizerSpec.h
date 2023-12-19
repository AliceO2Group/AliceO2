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

#ifndef STEER_DIGITIZERWORKFLOW_EMCALDIGITIZER_H_
#define STEER_DIGITIZERWORKFLOW_EMCALDIGITIZER_H_

#include <memory>
#include <vector>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsEMCAL/Digit.h"
#include "EMCALBase/Hit.h"
#include "EMCALSimulation/Digitizer.h"
#include "EMCALSimulation/SDigitizer.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <DetectorsBase/BaseDPLDigitizer.h>

class TChain;

namespace o2
{

namespace ctp
{
class CTPConfiguration;
}

namespace emcal
{
class CalibLoader;

/// \brief Create new digitizer spec
/// \return Digitizer spec

/// \class DigitizerSpec
/// \brief Task for EMCAL digitization in the data processing layer
/// \author Anders Garritt Knospe <anders.knospe@cern.ch>, University of Houston
/// \author Markus Fasel <markus.fasel@cern.ch> Oak Ridge National laboratory
/// \since Nov 12, 2018
class DigitizerSpec final : public o2::base::BaseDPLDigitizer, public o2::framework::Task
{
 public:
  using o2::base::BaseDPLDigitizer::init;
  /// \brief Constructor
  DigitizerSpec(std::shared_ptr<CalibLoader> calibloader, bool requireCTPInput) : o2::base::BaseDPLDigitizer(o2::base::InitServices::GEOM), o2::framework::Task(), mRequireCTPInput(requireCTPInput), mCalibHandler(calibloader) {}

  /// \brief Destructor
  ~DigitizerSpec() final = default;

  /// \brief init digitizer
  /// \param ctx Init context
  void initDigitizerTask(framework::InitContext& ctx) final;

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

  void configure();

  /// \brief run digitizer
  /// \param ctx Processing context
  ///
  /// Handling of pileup events:
  /// - Open readout window when the event sets a trigger
  /// - Accumulate digits sampled via the time response from different bunch crossings
  /// - Retrieve digits when the readout window closes
  void run(framework::ProcessingContext& ctx) override;

 private:
  Bool_t mFinished = false;                   ///< Flag for digitization finished
  bool mIsConfigured = false;                 ///< Initialization status of the digitizer
  bool mRunSDitizer = false;                  ///< Run SDigitization
  bool mRequireCTPInput = false;              ///< Require CTP min. bias input
  Digitizer mDigitizer;                       ///< Digitizer object
  o2::emcal::SDigitizer mSumDigitizer;        ///< Summed digitizer
  std::shared_ptr<CalibLoader> mCalibHandler; ///< Handler of calibration objects
  std::vector<Hit> mHits;                     ///< Vector with input hits
  std::vector<TChain*> mSimChains;
  o2::ctp::CTPConfiguration* mCTPConfig; ///< CTP configuration
};

/// \brief Create new digitizer spec
/// \return Digitizer spec
o2::framework::DataProcessorSpec getEMCALDigitizerSpec(int channel, bool requireCTPInput, bool mctruth = true, bool useccdb = true);

} // namespace emcal
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_EMCALDIGITIZER_H_ */
