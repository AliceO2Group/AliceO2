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
#include "DDSConfigHelpers.h"
#include "ChannelSpecHelpers.h"
#include <map>
#include <iostream>
#include <cstring>
#include <regex>
#include <fmt/format.h>
#include <libgen.h>

namespace o2::framework
{

struct ChannelProperties {
  std::string_view key;
  std::string_view value;
};

struct ChannelRewriter : FairMQChannelConfigParser {
  void beginChannel() override
  {
    names.push_back(-1);
    isZMQ.push_back(false);
    hasAddress.push_back(false);
    isWrite.push_back(false);
    propertiesBegin.push_back(propertyIndex);
  }

  void endChannel() override
  {
    propertiesEnd.push_back(propertyIndex);
    if (names.back() == -1) {
      throw std::runtime_error("Channel does not have a name.");
    }
    // If we have a zmq channel which does not have an address,
    // use a DDS property.
    if (isZMQ.back() && (hasAddress.back() == false)) {
      requiresProperties.push_back(channelIndex);
    }
    channelIndex++;
  }

  void property(std::string_view key, std::string_view value) override
  {
    properties.push_back({key, value});
    if (key == "address") {
      hasAddress.back() = true;
    }
    if (key == "transport" && value == "zeromq") {
      isZMQ.back() = true;
    }
    // Channels that bind need to write the bound address
    if (key == "method" && value == "bind") {
      isWrite.back() = true;
    }
    if (key == "name") {
      names.back() = propertyIndex;
    }
    propertyIndex++;
  }

  int propertyIndex = 0;
  int channelIndex = 0;
  std::vector<ChannelProperties> properties;
  std::vector<int> names;
  std::vector<int> requiresProperties;
  std::vector<int> propertiesBegin;
  std::vector<int> propertiesEnd;
  std::vector<bool> isWrite;
  std::vector<bool> isZMQ;
  std::vector<bool> hasAddress;
};

void dumpDeviceSpec2DDS(std::ostream& out,
                        std::string const& workflowSuffix,
                        const std::vector<DeviceSpec>& specs,
                        const std::vector<DeviceExecution>& executions,
                        const CommandInfo& commandInfo)
{
  out << R"(<topology name="o2-dataflow">)"
         "\n";
  assert(specs.size() == executions.size());
  std::vector<ChannelRewriter> rewriters;
  rewriters.resize(specs.size());

  // Find out if we need properties
  // and a property for each zmq channel which does not have
  // and address.
  for (size_t di = 0; di < specs.size(); ++di) {
    auto& rewriter = rewriters[di];
    auto& execution = executions[di];
    for (size_t cci = 0; cci < execution.args.size(); cci++) {
      const char* arg = execution.args[cci];
      if (!arg) {
        break;
      }
      if (strcmp(arg, "--channel-config") == 0) {
        if (cci + 1 == execution.args.size()) {
          throw std::runtime_error("wrong channel config found");
        }
        ChannelSpecHelpers::parseChannelConfig(execution.args[cci + 1], rewriter);
      }
    }
    for (int ci : rewriter.requiresProperties) {
      out << "   "
          << fmt::format("<property name=\"fmqchan_{}\" />\n", rewriter.properties[rewriter.names[ci]].value);
    }
  }

  float timeout = 0.0;

  for (size_t di = 0; di < specs.size(); ++di) {
    auto& spec = specs[di];
    auto& execution = executions[di];
    if (execution.args.empty()) {
      continue;
    }

    out << "   "
        << fmt::format("<decltask name=\"{}{}\">\n", spec.id, workflowSuffix);
    out << "       "
        << R"(<exe reachable="true">)";
    static bool doSleep = !getenv("DPL_DDS_SLEEP") || atoi(getenv("DPL_DDS_SLEEP"));
    if (doSleep) {
      out << fmt::format("sleep {}; ", timeout);
    }
    out << std::regex_replace(commandInfo.command, std::regex{"--dds(?!-)"}, "--dump") << " | ";
    timeout += 0.2;
    for (auto ei : execution.environ) {
      out << fmt::format(ei,
                         fmt::arg("timeslice0", spec.inputTimesliceId),
                         fmt::arg("timeslice1", spec.inputTimesliceId + 1),
                         fmt::arg("timeslice4", spec.inputTimesliceId + 4))
          << " ";
    }
    std::string accumulatedChannelPrefix;
    char* s = strdup(execution.args[0]);
    out << basename(s) << " ";
    free(s);
    for (size_t ai = 1; ai < execution.args.size(); ++ai) {
      const char* arg = execution.args[ai];
      if (!arg) {
        break;
      }
      if (strcmp(arg, "--id") == 0 && ai + 1 < execution.args.size()) {
        out << fmt::format(R"(--id {}_dds%TaskIndex%_%CollectionIndex% )", execution.args[ai + 1]);
        ai++;
        continue;
      }
      // Do not print out the driver client explicitly
      if (strcmp(arg, "--driver-client-backend") == 0) {
        ai++;
        continue;
      }
      if (strcmp(arg, "--control") == 0) {
        ai++;
        continue;
      }
      if (strcmp(arg, "--channel-prefix") == 0 &&
          ai + 1 < execution.args.size() &&
          *execution.args[ai + 1] == 0) {
        ai++;
        continue;
      }
      if (strpbrk(arg, "' ;@") != nullptr || arg[0] == 0) {
        out << fmt::format(R"("{}" )", arg);
      } else if (strpbrk(arg, "\"") != nullptr || arg[0] == 0) {
        out << fmt::format(R"('{}' )", arg);
      } else {
        out << fmt::format(R"({} )", arg);
      }
    }
    out << "--plugin odc";
    if (accumulatedChannelPrefix.empty() == false) {
      out << " --channel-config \"" << accumulatedChannelPrefix << "\"";
    }
    out << "</exe>\n";
    auto& rewriter = rewriters[di];
    if (rewriter.requiresProperties.empty() == false) {
      out << "   <properties>\n";
      for (auto pi : rewriter.requiresProperties) {
        out << fmt::format(
          "      <name access=\"{}\">fmqchan_{}</name>\n",
          rewriter.isWrite[pi] ? "write" : "read",
          rewriter.properties[rewriter.names[pi]].value);
      }
      out << "   </properties>\n";
    }
    out << "   </decltask>\n";
  }
  out << "   <declcollection name=\"DPL\">\n       <tasks>\n";
  for (const auto& spec : specs) {
    out << fmt::format("          <name>{}{}</name>\n", spec.id, workflowSuffix);
  }
  out << "       </tasks>\n   </declcollection>\n";
  out << "</topology>\n";
}

} // namespace o2::framework
