#include "CompareClusters.h"

#include <gpucf/ClusterChecker.h>
#include <gpucf/DataSet.h>


using namespace gpucf;


CompareClusters::CompareClusters()
    : Executable("Compare output of cluster finder against ground truth.")
{
}

void CompareClusters::setupFlags(
        args::Group &required, 
        args::Group &/*optional*/)
{
    truthFile = OptStringFlag( new StringFlag(
                    required, 
                    "FILE", 
                    "File with ground truth data.", 
                    {'t', "truth"}));

    clusterFile = OptStringFlag( new StringFlag(
                    required,
                    "FILE",
                    "Cluster file.",
                    {'c', "clusters"}));
}

int CompareClusters::mainImpl()
{
   DataSet truthSet;
   truthSet.read(truthFile->Get());
   std::vector<Cluster> truth = truthSet.deserialize<Cluster>();

   DataSet clusterSet;
   clusterSet.read(clusterFile->Get());
   std::vector<Cluster> clusters = clusterSet.deserialize<Cluster>();

   ClusterChecker cc(truth);
   cc.verify(clusters);

   return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:

