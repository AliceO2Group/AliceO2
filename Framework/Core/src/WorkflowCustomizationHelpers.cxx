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

#include "Framework/WorkflowCustomizationHelpers.h"
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <fmt/format.h>

namespace
{
std::string defaultIPCFolder()
{
#ifdef __linux__
  char const* channelPrefix = getenv("ALIEN_PROC_ID");
  if (channelPrefix) {
    return fmt::format("@dpl_{}_", channelPrefix);
  }
  return "@";
#else
  /// Find out a place where we can write the sockets
  char const* channelPrefix = getenv("TMPDIR");
  if (channelPrefix) {
    return {channelPrefix};
  }
  return access("/tmp", W_OK) == 0 ? "/tmp" : ".";
#endif
}
} // namespace

namespace o2::framework
{

std::vector<ConfigParamSpec> WorkflowCustomizationHelpers::requiredWorkflowOptions()
{
  return {{{"readers", VariantType::Int64, 1ll, {"number of parallel readers to use"}},
           {"spawners", VariantType::Int64, 1ll, {"number of parallel spawners to use"}},
           {"pipeline", VariantType::String, "", {"override default pipeline size"}},
           {"clone", VariantType::String, "", {"clone processors from a template"}},
           {"labels", VariantType::String, "", {"add labels to dataprocessors"}},
           {"workflow-suffix", VariantType::String, "", {"suffix to add to all dataprocessors"}},

           // options for TF rate limiting
           {"timeframes-rate-limit-ipcid", VariantType::String, "-1", {"Suffix for IPC channel for metric-feedback, -1 = disable"}},

           // options for AOD rate limiting
           {"aod-memory-rate-limit", VariantType::Int64, 0LL, {"Rate limit AOD processing based on memory"}},

           // options for AOD writer
           {"aod-writer-json", VariantType::String, "", {"Name of the json configuration file"}},
           {"aod-writer-resdir", VariantType::String, "", {"Name of the output directory"}},
           {"aod-writer-resfile", VariantType::String, "", {"Default name of the output file"}},
           {"aod-writer-maxfilesize", VariantType::Float, 0.0f, {"Maximum size of an output file in megabytes"}},
           {"aod-writer-resmode", VariantType::String, "RECREATE", {"Creation mode of the result files: NEW, CREATE, RECREATE, UPDATE"}},
           {"aod-writer-ntfmerge", VariantType::Int, -1, {"Number of time frames to merge into one file"}},
           {"aod-writer-keep", VariantType::String, "", {"Comma separated list of ORIGIN/DESCRIPTION/SUBSPECIFICATION:treename:col1/col2/..:filename"}},

           {"fairmq-rate-logging", VariantType::Int, 0, {"Rate logging for FairMQ channels"}},
           {"fairmq-recv-buffer-size", VariantType::Int, 4, {"recvBufferSize option for FairMQ channels"}},
           {"fairmq-send-buffer-size", VariantType::Int, 4, {"sendBufferSize option for FairMQ channels"}},
           /// Find out a place where we can write the sockets
           {"fairmq-ipc-prefix", VariantType::String, defaultIPCFolder(), {"Prefix for FairMQ channels location"}},

           {"forwarding-policy", VariantType::String, "dangling", {"Which messages to forward."
                                                                   " *dangling*: dangling outputs,"
                                                                   " all: all messages,"
                                                                   " none: no forwarding - it will complain if you try to create dangling outputs"}},
           {"forwarding-destination",
            VariantType::String,
            "drop",
            {"Destination for forwarded messages."
             " drop: simply drop them,"
             " file: write to file,"
             " fairmq: send to output proxy"}}}};
}
} // namespace o2::framework
