// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_EMCALDIGITIZER_H_
#define STEER_DIGITIZERWORKFLOW_EMCALDIGITIZER_H_

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
namespace emcal
{
/// \brief Create new digitizer spec
/// \return Digitizer spec

/// \class DigitizerSpec
/// \brief Task for EMCAL digitization in the data processing layer
/// \author Anders Garritt Knospe <anders.knospe@cern.ch>, University of Houston
/// \author Markus Fasel <markus.fasel@cern.ch> Oak Ridge National laboratory
/// \since Nov 12, 2018
class DigitizerSpec final : public o2::base::BaseDPLDigitizer
{
 public:
  /// \brief Constructor
  DigitizerSpec() : o2::base::BaseDPLDigitizer(o2::base::InitServices::GEOM) {}

  /// \brief Destructor
  ~DigitizerSpec() final = default;

  /// \brief init digitizer
  /// \param ctx Init context
  void initDigitizerTask(framework::InitContext& ctx) final;

  /// \brief run digitizer
  /// \param ctx Processing context
  ///
  /// Handling of pileup events:
  /// - Open readout window when the event sets a trigger
  /// - Accumulate digits sampled via the time response from different bunch crossings
  /// - Retrieve digits when the readout window closes
  void run(framework::ProcessingContext& ctx);

 private:
  Bool_t mFinished = false;            ///< Flag for digitization finished
  Digitizer mDigitizer;                ///< Digitizer object
  o2::emcal::SDigitizer mSumDigitizer; ///< Summed digitizer
  std::vector<TChain*> mSimChains;
  std::vector<Hit> mHits;                ///< Vector with input hits
  std::vector<Digit> mDigits;            ///< Vector with non-accumulated digits (per collision)
  std::vector<Digit> mAccumulatedDigits; ///< Vector with accumulated digits (time frame)
  dataformats::MCTruthContainer<MCLabel> mLabels;
};

/// \brief Create new digitizer spec
/// \return Digitizer spec
o2::framework::DataProcessorSpec getEMCALDigitizerSpec(int channel, bool mctruth = true);

} // end namespace emcal
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_EMCALDIGITIZER_H_ */
