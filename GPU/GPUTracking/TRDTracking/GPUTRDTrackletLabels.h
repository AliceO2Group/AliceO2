#ifndef GPUTRDTRACKLETLABELS_H
#define GPUTRDTRACKLETLABELS_H

namespace GPUCA_NAMESPACE
{
namespace gpu
{

// struct to hold the MC labels for the tracklets
struct GPUTRDTrackletLabels {
  int mLabel[3];
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
