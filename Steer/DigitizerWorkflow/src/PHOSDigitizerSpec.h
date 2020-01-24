// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_PHOSDIGITIZER_H_
#define STEER_DIGITIZERWORKFLOW_PHOSDIGITIZER_H_
#include <vector>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "PHOSBase/Hit.h"
#include "PHOSSimulation/Digitizer.h"
#include "SimulationDataFormat/MCTruthContainer.h"

class TChain;

namespace o2
{
namespace phos
{
/// \class DigitizerSpec
/// \brief Task for PHOS digitization in the data processing layer
/// \author Dmitri Peresunko, NRC "Kurchatov institute"
/// \author Adopted from EMCAL code write by Markus Fasel
/// \since Dec, 2019
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
  /// \param brname Name of the hit branch
  /// \param sourceID ID of the source
  /// \param entryID ID of the entry in the source
  void retrieveHits(const char* brname,
                    int sourceID,
                    int entryID);

  Bool_t mFinished = false;                                 ///< Flag for digitization finished
  Digitizer mDigitizer;                                     ///< Digitizer object
  TChain* mSimChain = nullptr;                              ///< Chain of files with background events
  TChain* mSimChainS = nullptr;                             ///< Chain of files with signal events
  std::vector<Hit>* mHitsBg = nullptr;                      ///< Vector with input hits from Bg event
  std::vector<Hit>* mHitsS = nullptr;                       ///< Vector with input hits from Signal event
  std::vector<Digit> mDigits;                               ///< Vector with non-accumulated digits (per collision)
  dataformats::MCTruthContainer<o2::phos::MCLabel> mLabels; ///< List of labels
};

/// \brief Create new digitizer spec
/// \return Digitizer spec
o2::framework::DataProcessorSpec getPHOSDigitizerSpec(int channel);

} // end namespace phos
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_PHOSDIGITIZER_H_ */
