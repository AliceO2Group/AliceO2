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
#ifndef O2_DIMESSAGES_H
#define O2_DIMESSAGES_H

#include <string>
#include <cstdint>
#include <vector>
#include <boost/optional.hpp>

namespace DIMessages
{
struct RegisterDevice {
  struct Specs {
    struct Input {
      std::string binding;
      std::string sourceChannel;
      size_t timeslice;

      boost::optional<std::string> origin;
      boost::optional<std::string> description;
      boost::optional<uint32_t> subSpec;
    };

    struct Output {
      std::string binding;
      std::string channel;
      size_t timeslice;
      size_t maxTimeslices;

      std::string origin;
      std::string description;
      boost::optional<uint32_t> subSpec;
    };

    struct Forward {
      std::string binding;
      size_t timeslice;
      size_t maxTimeslices;
      std::string channel;

      boost::optional<std::string> origin;
      boost::optional<std::string> description;
      boost::optional<uint32_t> subSpec;
    };

    std::vector<Input> inputs;
    std::vector<Output> outputs;
    std::vector<Forward> forwards;

    size_t rank;
    size_t nSlots;
    size_t inputTimesliceId;
    size_t maxInputTimeslices;
  };

  std::string name;
  std::string runId;
  Specs specs;

  std::string toJson();
};
} // namespace DIMessages

#endif // O2_DIMESSAGES_H
