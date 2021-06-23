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
#include "DetectorsBase/BaseDPLDigitizer.h"

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
  /// \brief helper function which will be offered as a service
  /// \param brname Name of the hit branch
  /// \param sourceID ID of the source
  /// \param entryID ID of the entry in the source
  void retrieveHits(const char* brname,
                    int sourceID,
                    int entryID);

  float mReadoutTime = 0.;                                  ///< PHOS readout time
  float mDeadTime = 0.;                                     ///< PHOS dead time
  Digitizer mDigitizer;                                     ///< Digitizer object
  std::vector<TChain*> mSimChains;                          ///< Chain of files for background/signal events
  std::vector<Hit>* mHits = nullptr;                        ///< Vector with input hits from Signal event
  std::vector<Digit> mDigitsTmp;                            ///< Vector with accumulated digits (per collision)
  std::vector<Digit> mDigitsFinal;                          ///< Vector with accumulated digits (per collision)
  std::vector<Digit> mDigitsOut;                            ///< Vector with accumulated digits (per collision)
  dataformats::MCTruthContainer<o2::phos::MCLabel> mLabels; ///< List of labels
};

/// \brief Create new digitizer spec
/// \return Digitizer spec
o2::framework::DataProcessorSpec getPHOSDigitizerSpec(int channel, bool mctruth = true);

} // end namespace phos
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_PHOSDIGITIZER_H_ */
