// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_EMCALDIGITWRITER_H_
#define STEER_DIGITIZERWORKFLOW_EMCALDIGITWRITER_H_

#include <memory> // for make_shared, make_unique, unique_ptr
#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>

#include "DataFormatsEMCAL/Digit.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace emcal
{

/// \class DigitsWriterSpec
/// \brief Task for EMCAL digits writer within the data processing layer
/// \author Anders Garritt Knospe <anders.knospe@cern.ch>, University of Houston
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since Nov 12, 2018
class DigitsWriterSpec : public framework::Task
{
 public:
  /// \brief Constructor
  DigitsWriterSpec() = default;

  /// \brief Destructor
  ~DigitsWriterSpec() final = default;

  /// \brief Init the digits writer
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Write digits and labels to the output file
  /// \param ctx processing context
  void run(framework::ProcessingContext& ctx) final;

 private:
  bool mFinished = false;                                 ///< flag indicating whether work is completed
  std::shared_ptr<TFile> mOutputFile;                     ///< Common output file
  std::shared_ptr<TTree> mOutputTree;                     ///< Common output tree
  std::shared_ptr<std::vector<o2::emcal::Digit>> mDigits; ///< Container for incoming digits (Sink responsible for deleting the digits)
};

/// \brief Create new digits writer spec
/// \return digits writer spec
o2::framework::DataProcessorSpec getEMCALDigitWriterSpec();

} // end namespace emcal
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_EMCALDIGITWRITER_H_ */
