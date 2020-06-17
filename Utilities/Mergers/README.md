<!-- doxy
\page refUtilitiesMergers Mergers
/doxy -->

# O2 Mergers

Mergers are DPL devices able to merge objects produced in parallel. These can be TObjects or custom object inheriting MergeInterface. Topologies of mergers can be created using the
`o2::mergers::MergerInfrastructureBuilder` class. To generate only one Merger, one can also use `o2::mergers::MergerBuilder`.
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

using namespace o2::mergers;

MergerInfrastructureBuilder mergersBuilder;

mergersBuilder.setInfrastructureName("histos");
mergersBuilder.setInputSpecs(mergersInputs);
mergersBuilder.setOutputSpec({{ "main" }, "TST", "HISTO", 0 });

MergerConfig config;
config.inputObjectTimespan = { InputObjectsTimespan::LastDifference };
config.mergedObjectTimespan = {MergedObjectTimespan::FullHistory};
config.publicationDecision = { PublicationDecision::EachNSeconds, 5};
config.topologySize = { TopologySize::NumberOfLayers, 2};

mergersBuilder.setConfig(config);

mergersBuilder.generateInfrastructure(specs); 
    
...
```

It creates a 2-layer topology of Mergers, which will consume `mergerInputs` and send merged object on the Output 
`{{"main"}, "TST", "HISTO", 0 }`. The infrastructure will integrate the received differences and each 5 seconds it will
 merge and publish the merged object. It will consist of a full history of the data that the topology will have received.