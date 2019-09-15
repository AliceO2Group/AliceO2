// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "CompareClusters.h"

#include <gpucf/common/DataSet.h>
#include <gpucf/debug/ClusterChecker.h>


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

