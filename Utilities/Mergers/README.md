<!-- doxy
\page refUtilitiesMergers Mergers
/doxy -->

# O2 Mergers (experimental)

Mergers are DPL devices able to merge ROOT objects produced in parallel. Topologies of mergers can be created using the
`o2::experimental::mergers::MergerInfrastructureBuilder` class. To generate only one Merger, one can also use `o2::experimental::mergers::MergerBuilder`.
Mergers provide a handful of options which are listed in the `include/Mergers/MergerConfig.h` file.

See the snippet from `src/mergersTopologyExample.cxx` as a usage example:
```cpp
...

void customize(std::vector<CompletionPolicy>& policies)
{
  MergerBuilder::customizeInfrastructure(policies);
}

#include <Framework/runDataProcessing.h>

...

using namespace o2::experimental::mergers;

MergerInfrastructureBuilder mergersBuilder;

mergersBuilder.setInfrastructureName("histos");
mergersBuilder.setInputSpecs(mergersInputs);
mergersBuilder.setOutputSpec({{ "main" }, "TST", "HISTO", 0 });

MergerConfig config;
config.ownershipMode = { OwnershipMode::Integral };
config.publicationDecision = { PublicationDecision::EachNSeconds, 5};
config.mergingTime = { MergingTime::BeforePublication};
config.timespan = {Timespan::FullHistory};
config.topologySize = { TopologySize::NumberOfLayers, 2};

mergersBuilder.setConfig(config);

mergersBuilder.generateInfrastructure(specs); 
    
...
```

It creates a 2-layer topology of Mergers, which will consume `mergerInputs` and send merged object on the Output 
`{{"main"}, "TST", "HISTO", 0 }`. The infrastructure will integrate the received differences and each 5 seconds it will
 merge and publish the merged object. It will consist of a full history of the data that topology will have received.