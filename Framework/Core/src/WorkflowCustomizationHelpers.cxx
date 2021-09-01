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

namespace
{
std::string defaultIPCFolder()
{
  /// Find out a place where we can write the sockets
  char const* channelPrefix = getenv("TMPDIR");
  if (channelPrefix) {
    return std::string(channelPrefix);
  }
  return access("/tmp", W_OK) == 0 ? "/tmp" : ".";
}
} // namespace

namespace o2::framework
{

std::vector<ConfigParamSpec> WorkflowCustomizationHelpers::requiredWorkflowOptions()
{
  return std::vector<ConfigParamSpec>{{ConfigParamSpec{"readers", VariantType::Int64, 1ll, {"number of parallel readers to use"}},
                                       ConfigParamSpec{"spawners", VariantType::Int64, 1ll, {"number of parallel spawners to use"}},
                                       ConfigParamSpec{"pipeline", VariantType::String, "", {"override default pipeline size"}},
                                       ConfigParamSpec{"clone", VariantType::String, "", {"clone processors from a template"}},
                                       ConfigParamSpec{"labels", VariantType::String, "", {"add labels to dataprocessors"}},
                                       ConfigParamSpec{"workflow-suffix", VariantType::String, "", {"suffix to add to all dataprocessors"}},

                                       // options for AOD rate limiting
                                       ConfigParamSpec{"aod-memory-rate-limit", VariantType::Int64, 0LL, {"Rate limit AOD processing based on memory"}},

                                       // options for AOD writer
                                       ConfigParamSpec{"aod-writer-json", VariantType::String, "", {"Name of the json configuration file"}},
                                       ConfigParamSpec{"aod-writer-resfile", VariantType::String, "", {"Default name of the output file"}},
                                       ConfigParamSpec{"aod-writer-resmode", VariantType::String, "RECREATE", {"Creation mode of the result files: NEW, CREATE, RECREATE, UPDATE"}},
                                       ConfigParamSpec{"aod-writer-ntfmerge", VariantType::Int, -1, {"Number of time frames to merge into one file"}},
                                       ConfigParamSpec{"aod-writer-keep", VariantType::String, "", {"Comma separated list of ORIGIN/DESCRIPTION/SUBSPECIFICATION:treename:col1/col2/..:filename"}},

                                       ConfigParamSpec{"fairmq-rate-logging", VariantType::Int, 0, {"Rate logging for FairMQ channels"}},
                                       ConfigParamSpec{"fairmq-recv-buffer-size", VariantType::Int, 4, {"recvBufferSize option for FairMQ channels"}},
                                       ConfigParamSpec{"fairmq-send-buffer-size", VariantType::Int, 4, {"sendBufferSize option for FairMQ channels"}},
                                       /// Find out a place where we can write the sockets
                                       ConfigParamSpec{"fairmq-ipc-prefix", VariantType::String, defaultIPCFolder(), {"Prefix for FairMQ channels location"}},

                                       ConfigParamSpec{"forwarding-policy", VariantType::String, "dangling", {"Which messages to forward."
                                                                                                              " *dangling*: dangling outputs,"
                                                                                                              " all: all messages,"
                                                                                                              " none: no forwarding - it will complain if you try to create dangling outputs"}},
                                       ConfigParamSpec{"forwarding-destination",
                                                       VariantType::String,
                                                       "drop",
                                                       {"Destination for forwarded messages."
                                                        " drop: simply drop them,"
                                                        " file: write to file,"
                                                        " fairmq: send to output proxy"}}}};
}
} // namespace o2::framework
