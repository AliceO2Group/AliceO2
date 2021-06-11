// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FITDigitWriterSpec.h

#ifndef O2_FITDIGITWRITER_H
#define O2_FITDIGITWRITER_H

#include <Framework/Logger.h>
#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include <tuple>
#include <vector>
#include <string>

using namespace o2::framework;

namespace o2
{
namespace fit
{
template <typename RawReaderType, typename MCLabelContainerType /*For example: =o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>*/>
struct FITDigitWriterSpecHelper {
  typedef RawReaderType RawReader_t;
  typedef MCLabelContainerType MCLabelContainer_t; //should be defined as type field within Digit structure
  typedef typename RawReader_t::Digit_t Digit_t;
  typedef typename Digit_t::DetTrigInput_t DetTrigInput_t;
  typedef typename RawReader_t::SubDigit_t SubDigit_t;             //tuple of vectors
  typedef typename RawReader_t::SingleSubDigit_t SingleSubDigit_t; //tuple of vectors
  typedef typename RawReader_t::IndexesSubDigit IndexesSubDigit_t;
  typedef typename RawReader_t::IndexesSingleSubDigit IndexesSingleSubDigit_t;

  template <typename T>
  using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

  static o2::framework::DataProcessorSpec getFITDigitWriterSpec(bool mctruth, bool trigInp, o2::header::DataOrigin dataOrigin)
  {
    if (trigInp) {
      return callMakeSpec<true>(mctruth, dataOrigin, IndexesSubDigit_t{}, IndexesSingleSubDigit_t{});
    } else {
      return callMakeSpec<false>(mctruth, dataOrigin, IndexesSubDigit_t{}, IndexesSingleSubDigit_t{});
    }
  }

  template <typename T, typename... Args>
  static auto getBranchDef(o2::header::DataOrigin dataOrigin, Args&&... args)
  {
    std::string detName = dataOrigin.template as<std::string>();
    std::string detNameLower = detName;
    std::for_each(detNameLower.begin(), detNameLower.end(), [](char& c) { c = ::tolower(c); });
    auto dplName = T::sChannelNameDPL;
    auto dplLabel = std::string{dplName};
    std::for_each(dplLabel.begin(), dplLabel.end(), [](char& c) { c = ::tolower(c); });
    auto branchName = std::string{detName + dplName};
    auto optionStr = std::string{detNameLower + "-" + dplName + "-branch-name"};
    //LOG(INFO)<<"Branch: "<<dplLabel.c_str()<< "|" <<detName<<" | "<<T::sChannelNameDPL<<" | "<<branchName<<" | "<<optionStr<<" | "<<(detName+dplName);
    return BranchDefinition<std::vector<T>>{InputSpec{dplLabel.c_str(), dataOrigin, T::sChannelNameDPL}, branchName.c_str(), optionStr.c_str(), std::forward<Args>(args)...};
  }

  template <std::size_t N, typename TupleType, typename... Args>
  static auto getBranchDefFromTuple(o2::header::DataOrigin dataOrigin, Args&&... args)
  {
    using ObjType = typename std::tuple_element<N, TupleType>::type::value_type;
    return getBranchDef<ObjType>(dataOrigin, std::forward<Args>(args)...);
  }

  template <bool trigInp, std::size_t... IsubDigits, std::size_t... IsingleSubDigits>
  static auto callMakeSpec(bool mctruth, o2::header::DataOrigin dataOrigin, std::index_sequence<IsubDigits...>, std::index_sequence<IsingleSubDigits...>)
  {
    using InputSpec = framework::InputSpec;
    using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;

    std::string detName = dataOrigin.template as<std::string>();
    std::string detNameLower = detName;
    std::for_each(detNameLower.begin(), detNameLower.end(), [](char& c) { c = ::tolower(c); });
    // Spectators for logging
    auto logger = [](std::vector<Digit_t> const& vecDigits) {
      LOG(INFO) << "FITDigitWriter pulled " << vecDigits.size() << " digits";
    };
    // the callback to be set as hook for custom action when the writer is closed
    auto finishWriting = [](TFile* outputfile, TTree* outputtree) {
      const auto* brArr = outputtree->GetListOfBranches();
      int64_t nent = 0;
      for (const auto* brc : *brArr) {
        int64_t n = ((const TBranch*)brc)->GetEntries();
        if (nent && (nent != n)) {
          LOG(ERROR) << "Branches have different number of entries";
        }
        nent = n;
      }
      outputtree->SetEntries(nent);
      outputtree->Write();
      outputfile->Close();
    };
    auto digitsdef = getBranchDef<Digit_t>(dataOrigin, 1, logger);
    auto trginputdef = getBranchDef<DetTrigInput_t>(dataOrigin);
    auto labelsdef = BranchDefinition<MCLabelContainer_t>{InputSpec{"labelinput", dataOrigin, "DIGITSMCTR"}, std::string{detName + "DIGITSMCTR"}.c_str(), mctruth ? 1 : 0};
    if constexpr (trigInp == false) {
      return MakeRootTreeWriterSpec(
        std::string{detName + "DigitWriterRaw"}.c_str(),
        std::string{"o2_" + detNameLower + "digits.root"}.c_str(),
        "o2sim",
        MakeRootTreeWriterSpec::CustomClose(finishWriting),
        getBranchDefFromTuple<IsubDigits, SubDigit_t>(dataOrigin)...,
        getBranchDefFromTuple<IsingleSubDigits, SingleSubDigit_t>(dataOrigin)...,
        std::move(digitsdef),
        std::move(labelsdef))();
    } else {
      return MakeRootTreeWriterSpec(
        std::string{detName + "DigitWriter"}.c_str(),
        std::string{detNameLower + "digits.root"}.c_str(),
        "o2sim",
        MakeRootTreeWriterSpec::CustomClose(finishWriting),
        getBranchDefFromTuple<IsubDigits, SubDigit_t>(dataOrigin)...,
        getBranchDefFromTuple<IsingleSubDigits, SingleSubDigit_t>(dataOrigin)...,
        std::move(digitsdef),
        std::move(trginputdef),
        std::move(labelsdef))();
    }
  }
};
} // namespace fit
} // namespace o2

#endif /* O2_FITDIGITWRITER_H */
