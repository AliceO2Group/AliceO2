#pragma once

#include <gpucf/algorithms/ClusterFinder.h>
#include <gpucf/debug/utils.h>


namespace gpucf
{
    
class PeakCountTest : public ClusterFinder
{
    
public:

   PeakCountTest(ClusterFinderConfig, ClEnv); 

   bool run(
           const Array2D<float> &, 
           const Array2D<unsigned char> &, 
           const Array2D<char> &);

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
