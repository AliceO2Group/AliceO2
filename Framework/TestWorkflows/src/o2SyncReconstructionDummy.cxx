// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"

using namespace o2::framework;

AlgorithmSpec simplePipe(std::string const& what)
{
  return AlgorithmSpec{ [what](InitContext& ic) {
    auto delay = ic.options().get<int>("delay");
    auto messageSize = ic.options().get<int>("size");

    return [what, delay, messageSize](ProcessingContext& ctx) {
      sleep(delay);
      ctx.outputs().make<char>(OutputRef{ what }, messageSize);
    };
  } };
}

// Helper to create two options, one for the delay between messages,
// another one for the size of the message.
std::vector<ConfigParamSpec> simplePipeOptions(int delay, int size)
{
  return {
    ConfigParamSpec{ "delay", VariantType::Int, 1, { "Delay between one iteration and the other" } },
    ConfigParamSpec{ "size", VariantType::Int, 1, { "Size of the output message" } },
  };
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const&specs) {
  WorkflowSpec workflow{
    DataProcessorSpec{ "flp-dpl-source",
      {},
      { OutputSpec{ { "tpc-cluster" }, "TPC", "CLUSTERS" },
        OutputSpec{ { "its-cluster" }, "ITS", "CLUSTERS" },
        OutputSpec{ { "trd-tracklet" }, "TRD", "TRACKLETS" },
        OutputSpec{ { "tof-cluster" }, "TOF", "CLUSTERS" },
        OutputSpec{ { "fit-data" }, "FIT", "DATA" },
        OutputSpec{ { "mch-cluster" }, "MCH", "CLUSTERS" },
        OutputSpec{ { "mid-cluster" }, "MID", "CLUSTERS" },
        OutputSpec{ { "emcal-cluster" }, "EMC", "CLUSTERS" },
        OutputSpec{ { "phos-cluster" }, "PHO", "CLUSTERS" },
        OutputSpec{ { "mft-cluster" }, "MFT", "CLUSTERS" },
        OutputSpec{ { "hmpid-cluster" }, "HMP", "CLUSTERS" }
      },
      AlgorithmSpec{
        [](InitContext &setup) {
          auto delay = setup.options().get<int>("epn-roundrobin-delay");
          return [delay](ProcessingContext& ctx) {
            sleep(delay);
            ctx.outputs().make<int>(OutputRef { "tpc-cluster" }, 1);
            ctx.outputs().make<int>(OutputRef { "its-cluster" }, 1);
            ctx.outputs().make<int>(OutputRef { "trd-tracklet" }, 1);
            ctx.outputs().make<int>(OutputRef { "tof-cluster" }, 1);
            ctx.outputs().make<int>(OutputRef { "fit-data" }, 1);
            ctx.outputs().make<int>(OutputRef { "mch-cluster" }, 1);
            ctx.outputs().make<int>(OutputRef { "mid-cluster" }, 1);
            ctx.outputs().make<int>(OutputRef { "emcal-cluster" }, 1);
            ctx.outputs().make<int>(OutputRef { "phos-cluster" }, 1);
            ctx.outputs().make<int>(OutputRef { "mft-cluster" }, 1);
            ctx.outputs().make<int>(OutputRef { "hmpid-cluster" }, 1);
          };
        }
      },
      {
        ConfigParamSpec{ "epn-roundrobin-delay", VariantType::Int, 28, { "Fake delay for waiting from the network for a new timeframe" } }
      }
    },
    DataProcessorSpec{ "tpc-tracking",
      { InputSpec{ "clusters", "TPC", "CLUSTERS" } },
      { OutputSpec{ { "tracks" }, "TPC", "TRACKS" } },
      simplePipe("tracks"),
      simplePipeOptions(25, 1)
    },
    DataProcessorSpec{ "its-tracking",
      { InputSpec{ "clusters", "ITS", "CLUSTERS" } },
      { OutputSpec{ { "tracks" }, "ITS", "TRACKS" } },
      simplePipe("tracks"),
      simplePipeOptions(25, 1)
    },
    DataProcessorSpec{ "its-cluster-compression",
      { InputSpec{ "clusters", "ITS", "CLUSTERS" } },
      { OutputSpec{ { "compressed-clusters" }, "ITS", "CLUSTERS_C" } },
      simplePipe("compressed-clusters"),
      simplePipeOptions(25, 1)
    },
    DataProcessorSpec{ "trd-tracklets-compression",
      { InputSpec {"tracklets", "TRD", "TRACKLETS" } },
      { OutputSpec {{"compressed"}, "TRD", "TRACKLETS_C" } },
      simplePipe("compressed"),
      simplePipeOptions(5, 1)
    },
    DataProcessorSpec{ "tof-cluster-compression",
      { InputSpec{ "clusters", "TOF", "CLUSTERS" } },
      { OutputSpec{ { "compressed-clusters" }, "TOF", "CLUSTERS_C" } },
      simplePipe("compressed-clusters"),
      simplePipeOptions(1, 1)
    },
    DataProcessorSpec{ "fit-reconstruction",
      { InputSpec{ "fit-data", "FIT", "DATA" } },
      { OutputSpec{ { "interaction-times" }, "FIT", "I_TIMES" } },
      simplePipe("interaction-times"),
      simplePipeOptions(1, 1)
    },
    DataProcessorSpec{ "mch-cluster-compression",
      { InputSpec{ "clusters", "MCH", "CLUSTERS" } },
      { OutputSpec{ { "compressed-clusters" }, "MCH", "CLUSTERS_C" } },
      simplePipe("compressed-clusters"),
      simplePipeOptions(1, 1)
    },
    DataProcessorSpec{ "mid-cluster-compression",
      { InputSpec{ "clusters", "MID", "CLUSTERS" } },
      { OutputSpec{ { "compressed-clusters" }, "MID", "CLUSTERS_C" } },
      simplePipe("compressed-clusters"),
      simplePipeOptions(1, 1)
    },
    DataProcessorSpec{ "emc-cluster-compression",
      { InputSpec{ "clusters", "EMC", "CLUSTERS" } },
      { OutputSpec{ { "compressed-clusters" }, "EMC", "CLUSTERS_C" } },
      simplePipe("compressed-clusters"),
      simplePipeOptions(1, 1)
    },
    DataProcessorSpec{ "phos-cluster-compression",
      { InputSpec{ "clusters", "PHO", "CLUSTERS" } },
      { OutputSpec{ { "compressed-clusters" }, "PHO", "CLUSTERS_C" } },
      simplePipe("compressed-clusters"),
      simplePipeOptions(1, 1)
    },
    DataProcessorSpec{ "mft-cluster-compression",
      { InputSpec{ "clusters", "MFT", "CLUSTERS" } },
      { OutputSpec{ { "compressed-clusters" }, "MFT", "CLUSTERS_C" } },
      simplePipe("compressed-clusters"),
      simplePipeOptions(1, 1)
    },
    DataProcessorSpec{ "hmpid-cluster-compression",
      { InputSpec{ "clusters", "HMP", "CLUSTERS" } },
      { OutputSpec{ { "compressed-clusters" }, "HMP", "CLUSTERS_C" } },
      simplePipe("compressed-clusters"),
      simplePipeOptions(1, 1)
    },
    DataProcessorSpec{
      "its-tpc-matching",
      {
        InputSpec{"tpc-tracks", "TPC", "TRACKS"},
        InputSpec{"its-tracks", "ITS", "TRACKS"}
      },
      {
        OutputSpec{{"tpc-its-matching"}, "MTC", "TPC_ITS"}
      },
      simplePipe("tpc-its-matching"),
      simplePipeOptions(3, 1)
    },
    DataProcessorSpec{
      "its-tpc-trd-tof-matching",
      {
        InputSpec{"tpc-tracks", "TPC", "TRACKS"},
        InputSpec{"its-tracks", "ITS", "TRACKS"},
        InputSpec{"tpc-its-matching", "MTC", "TPC_ITS"}
      },
      {
        OutputSpec{{"tpc-its-trd-tof-matching"}, "MTC", "ALL"}
      },
      simplePipe("tpc-its-trd-tof-matching"),
      simplePipeOptions(2, 1)
    },
    DataProcessorSpec{
      "tpc-compression",
      {InputSpec{"tpc-tracks", "TPC", "TRACKS"},
       InputSpec{"tpc-clusters", "TPC", "CLUSTERS"}},
      {
        OutputSpec{{"tpc-compressed"}, "TPC", "COMPRESSED"}
      },
      simplePipe("tpc-compressed"),
      simplePipeOptions(5, 1)
    },
    DataProcessorSpec{
      "tpc-sp-calibration",
      {
        InputSpec{"tpc-tracks", "TPC", "TRACKS"},
        InputSpec{"its-tracks", "ITS", "TRACKS"},
        InputSpec{"tpc-its-matching", "MTC", "TPC_ITS"},
        InputSpec{"tpc-its-matching", "MTC", "ALL"}
      },
      {
        OutputSpec{{"residuals"}, "SPC", "RESIDUALS"}
      },
      simplePipe("residuals"),
      simplePipeOptions(5, 1)
    },
    DataProcessorSpec{
      "writer",
      Inputs{
        InputSpec {"compressed-its", "ITS", "CLUSTERS_C" },
        InputSpec {"compressed-fit", "FIT", "I_TIMES" },
        InputSpec {"compressed-trd", "TRD", "TRACKLETS_C" },
        InputSpec {"compressed-tof", "TOF", "CLUSTERS_C" },
        InputSpec {"compressed-mch", "MCH", "CLUSTERS_C" },
        InputSpec {"compressed-mid", "MID", "CLUSTERS_C" },
        InputSpec {"compressed-emc", "EMC", "CLUSTERS_C" },
        InputSpec {"compressed-pho", "PHO", "CLUSTERS_C" },
        InputSpec {"compressed-mft", "MFT", "CLUSTERS_C" },
        InputSpec {"compressed-hmp", "HMP", "CLUSTERS_C" },
        InputSpec {"compressed-res", "SPC", "RESIDUALS" },
        InputSpec {"compressed-tpc", "TPC", "COMPRESSED" }
      },
      {}
    }
  };
  return workflow;
}
