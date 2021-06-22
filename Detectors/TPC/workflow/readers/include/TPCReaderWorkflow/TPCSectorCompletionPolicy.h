// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_TPCSECTORCOMPLETIONPOLICY_H
#define O2_TPC_TPCSECTORCOMPLETIONPOLICY_H
/// @file   TPCSectorCompletionPolicy.h
/// @author Matthias Richter
/// @since  2020-05-20
/// @brief  DPL completion policy helper for TPC scetor data

#include "Framework/CompletionPolicy.h"
#include "Framework/InputSpec.h"
#include "Framework/InputSpan.h"
#include "Framework/DeviceSpec.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "Headers/DataHeaderHelpers.h"
#include "TPCBase/Sector.h"
#include <fmt/ostream.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <regex>

namespace o2
{
namespace tpc
{

/// @class TPCSectorCompletionPolicy
/// A helper class providing a DPL completion policy definition for the TPC sector data
///
/// The class can be used as a functor creating the CompletionPolicy object based on the parameters
/// provided to the constructor.
///
/// TPC data is split on the level of sectors. For processors requiring the full data set like e.g.
/// the tracker, DPL will synchronize complete data sets from the input routes. Multiple O2 messages
/// can be received on one input route, a custom completion policy is needed to define the complete
/// data set. All TPC data except raw data is sent with a specific TPCSectorHeader on the header
/// stack describing the current sector and defining the active sectors in the setup. The completion
/// policy callback will wait until there is data for all active sectors.
///
/// If config flag ConFig::RequireAll is specified in the constructor parameters, data from all inputs
/// will be required in addition to the matching TPC sector policy. With this flag, the policy can be
/// used for processors with TPC input and other inputs, without checking for complex multimessages on
/// the other inputs.
///
/// Parameters:
///   processor name   rexexp to match a name of the processor for which the policy should be applied
///   input matchers   provided as an argument pack
///                    Note: it is important to use ConcreteDataTypeMatcher to define input spec with
///                          wildcard on subSpecification
///   config param     Parameters like Config::RequireAll
/// Usage:
///   TPCSectorCompletionPolicy("processor-name-regexp",
///                             TPCSectorCompletionPolicy::Config::RequireAll,
///                             InputSpec{"", ConcreteDataTypeMatcher{"DET", "RAWDATA"}}, ...)();
///
class TPCSectorCompletionPolicy
{
 public:
  using CompletionPolicyData = std::vector<framework::InputSpec>;

  enum struct Config {
    // require data on all other inputs in addition to the ones checked for the sector completion
    RequireAll,
  };
  TPCSectorCompletionPolicy() = delete;
  template <typename... Args>
  TPCSectorCompletionPolicy(const char* processorName, Args&&... args)
    : mProcessorName(processorName), mInputMatchers()
  {
    init(std::forward<Args>(args)...);
  }

  o2::framework::CompletionPolicy operator()()
  {
    constexpr static size_t NSectors = o2::tpc::Sector::MAXSECTOR;

    auto matcher = [expression = mProcessorName](framework::DeviceSpec const& device) -> bool {
      return std::regex_match(device.name.begin(), device.name.end(), std::regex(expression.c_str()));
    };

    auto callback = [bRequireAll = mRequireAll, inputMatchers = mInputMatchers, externalInputMatchers = mExternalInputMatchers, pTpcSectorMask = mTpcSectorMask](framework::InputSpan const& inputs) -> framework::CompletionPolicy::CompletionOp {
      unsigned long tpcSectorMask = pTpcSectorMask ? *pTpcSectorMask : 0xFFFFFFFFF;
      std::bitset<NSectors> validSectors = 0;
      bool haveMatchedInput = false;
      uint64_t activeSectors = 0;
      std::vector<uint64_t> validSectorsExternal(externalInputMatchers ? externalInputMatchers->size() : 0);
      size_t nActiveInputRoutes = 0;
      size_t nMaxPartsPerRoute = 0;
      int inputType = -1;
      for (auto it = inputs.begin(), end = inputs.end(); it != end; ++it) {
        nMaxPartsPerRoute = it.size() > nMaxPartsPerRoute ? it.size() : nMaxPartsPerRoute;
        bool haveActivePart = false;
        for (auto const& ref : it) {
          if (!framework::DataRefUtils::isValid(ref)) {
            continue;
          }
          haveActivePart = true;
          auto const* dh = framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
          // check if the O2 message matches on of the input specs to be matched and
          // if it matches, check for the sector header, retrieve the active sector information
          // and mark the sector as valid, we require to match exactly one of the inputs in the list
          for (size_t idx = 0, end = inputMatchers.size(); idx < end; idx++) {
            auto const& spec = inputMatchers[idx];
            if (framework::DataRefUtils::match(ref, spec)) {
              haveMatchedInput = true;
              if (inputType == -1) {
                // we bind to the index of the first match and require all other inputs to match the same spec
                inputType = idx;
              } else if (inputType != idx) {
                std::stringstream error;
                error << fmt::format("routing error, input messages must all be of the same type previously bound to {} {}/{}/{}",
                                     inputMatchers[inputType],
                                     dh->dataOrigin,
                                     dh->dataDescription, dh->subSpecification);
                throw std::runtime_error(error.str());
              }
              auto const* sectorHeader = framework::DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
              if (sectorHeader == nullptr) {
                throw std::runtime_error("TPC sector header missing on header stack");
              }
              activeSectors |= (sectorHeader->activeSectors & tpcSectorMask);
              validSectors |= (sectorHeader->sectorBits & tpcSectorMask);
              break;
            }
          }

          // We require to match all inputs in the external list
          if (externalInputMatchers) {
            for (size_t idx = 0, end = externalInputMatchers->size(); idx < end; idx++) {
              auto const& spec = (*externalInputMatchers)[idx];
              if (framework::DataRefUtils::match(ref, spec)) {
                auto const* sectorHeader = framework::DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
                if (sectorHeader == nullptr) {
                  throw std::runtime_error("TPC sector header missing on header stack");
                }
                activeSectors |= (sectorHeader->activeSectors & tpcSectorMask);
                validSectorsExternal[idx] |= (sectorHeader->sectorBits & tpcSectorMask);
                break;
              }
            }
          }
        }
        if (haveActivePart) {
          ++nActiveInputRoutes;
        }
      }

      if (externalInputMatchers) {
        // We require all external matchers to have all sectors present, if not we wait
        for (size_t idx = 0, end = externalInputMatchers->size(); idx < end; idx++) {
          if (validSectorsExternal[idx] == 0 || validSectorsExternal[idx] != activeSectors) {
            return framework::CompletionPolicy::CompletionOp::Wait;
          }
        }
      }

      // If the flag Config::RequireAll is set in the constructor arguments we require
      // data from all inputs in addition to the sector matching condition
      // To be fully correct we would need to require data from all inputs not going
      // into the TPC policy, but that is not possible for the moment. That's why there is a possibly
      // unhandled case if multiple TPC input routes are defined but a complete data set is coming over
      // one of them. Not likely to be a use case, though.
      if ((inputMatchers.size() == 0 || (haveMatchedInput && activeSectors == validSectors.to_ulong())) &&
          (!bRequireAll || nActiveInputRoutes == inputs.size())) {
        // we can process if there is input for all sectors, the required sectors are
        // transported as part of the sector header
        return framework::CompletionPolicy::CompletionOp::Consume;
      } else if (activeSectors == 0 && nActiveInputRoutes == inputs.size()) {
        // no sector header is transmitted, this is the case for e.g. the ZS raw data
        // we simply require input on all routes, this is also the default of DPL DataRelayer
        // Because DPL can not do more without knowing how many parts are required for a complete
        // data set, the workflow should be such that exactly one O2 message arrives per input route.
        // Currently, the workflow has multiple O2 messages per input route, but they all come in
        // a single multipart message. So it works fine, and we disable the warning below, but there
        // is a potential problem. Need to fix this on the level of the workflow.
        //if (nMaxPartsPerRoute > 1) {
        //  LOG(WARNING) << "No sector information is provided with the data, data set is complete with data on all input routes. But there are multiple parts on at least one route and this policy might not be complete, no check possible if other parts on some routes are still missing. It is adviced to add a custom policy.";
        //}
        return framework::CompletionPolicy::CompletionOp::Consume;
      }

      return framework::CompletionPolicy::CompletionOp::Wait;
    };
    return framework::CompletionPolicy{"TPCSectorCompletionPolicy", matcher, callback};
  }

 private:
  /// recursively init list of input routes from parameter pack
  template <typename Arg, typename... Args>
  void init(Arg&& arg, Args&&... args)
  {
    using Type = std::decay_t<Arg>;
    if constexpr (std::is_same_v<Type, framework::InputSpec>) {
      mInputMatchers.emplace_back(std::move(arg));
    } else if constexpr (std::is_same_v<Type, TPCSectorCompletionPolicy::Config>) {
      switch (arg) {
        case Config::RequireAll:
          mRequireAll = true;
          break;
      }
    } else if constexpr (std::is_same_v<Type, std::vector<o2::framework::InputSpec>*>) {
      mExternalInputMatchers = arg;
    } else if constexpr (std::is_same_v<Type, unsigned long*> || std::is_same_v<Type, const unsigned long*>) {
      mTpcSectorMask = arg;
    } else {
      static_assert(framework::always_static_assert_v<Type>);
    }
    if constexpr (sizeof...(args) > 0) {
      init(std::forward<Args>(args)...);
    }
  }

  std::string mProcessorName;
  std::vector<framework::InputSpec> mInputMatchers;
  // The external input matchers behave as the internal ones with the following differences:
  // - They are controlled externally and the external entity can modify them, e.g. after parsing command line arguments.
  // - They are all matched independently, it is not sufficient that one of them is present for all sectors
  const std::vector<framework::InputSpec>* mExternalInputMatchers = nullptr;
  const unsigned long* mTpcSectorMask = nullptr;
  bool mRequireAll = false;
};
} // namespace tpc
} // namespace o2
#endif // O2_TPC_TPCSECTORCOMPLETIONPOLICY_H
