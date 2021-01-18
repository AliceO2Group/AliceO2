// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    DatDecoderSpec.h
/// \author  Andrea Ferrero
///
/// \brief Definition of a data processor to run the raw decoding
///

#ifndef O2_MCH_DATADECODERSPEC_H_
#define O2_MCH_DATADECODERSPEC_H_

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace hmpid
{

  class DataDecoderTask : public Task
  {
    public:
      DataDecoderTask() = default;
      ~DataDecoderTask() override = default;
      void init(framework::InitContext& ic) final;
      void run(framework::ProcessingContext& pc) final;
      void decodeTF(framework::ProcessingContext& pc);

    private:
      o2::hmpid::HmpidDecodeRawDigit *mDeco;
 //     vector<o2::hmpid::Digit> mDigits;

  };

o2::framework::DataProcessorSpec getDecodingSpec(std::string inputSpec = "TF:HMP/RAWDATA");
//o2::framework::DataProcessorSpec getDecodingSpec();
} // end namespace hmpid
} // end namespace o2

#endif
