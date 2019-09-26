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
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

class TChain;

namespace o2
{
namespace emcal
{

/// \class DigitizerSpec
/// \brief Task for EMCAL digitization in the data processing layer
/// \author Anders Garritt Knospe <anders.knospe@cern.ch>, University of Houston
/// \author Markus Fasel <markus.fasel@cern.ch> Oak Ridge National laboratory
/// \since Nov 12, 2018
class DigitizerSpec : public framework::Task
{
 public:
  /// \brief Constructor
  DigitizerSpec() = default;

  /// \brief Destructor
  ~DigitizerSpec() final = default;

  /// \brief init digitizer
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief run digitizer
  /// \param ctx Processing context
  ///
  /// Handling of pileup events:
  /// - Open readout window when the event sets a trigger
  /// - Accumulate digits sampled via the time response from different bunch crossings
  /// - Retrieve digits when the readout window closes
  void run(framework::ProcessingContext& ctx) final;

 private:
  /// \brief helper function which will be offered as a service
  /// \param chains Input chains
  /// \param brname Name of the hit branch
  /// \param sourceID ID of the source
  /// \param entryID ID of the entry in the source
  /// \param hits output vector of hits
  void retrieveHits(std::vector<TChain*> const& chains,
                    const char* brname,
                    int sourceID,
                    int entryID,
                    std::vector<Hit>* hits);

  Bool_t mFinished = false; ///< Flag for digitization finished
  Digitizer mDigitizer;     ///< Digitizer object
  std::vector<TChain*> mSimChains;
  std::vector<Hit> mHits;                ///< Vector with input hits
  std::vector<Digit> mDigits;            ///< Vector with non-accumulated digits (per collision)
  std::vector<Digit> mAccumulatedDigits; /// Vector with accumulated digits (time frame)
  dataformats::MCTruthContainer<o2::MCCompLabel> mLabels;
};

/// \brief Create new digitizer spec
/// \return Digitizer spec
o2::framework::DataProcessorSpec getEMCALDigitizerSpec(int channel);

} // end namespace emcal
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_EMCALDIGITIZER_H_ */
