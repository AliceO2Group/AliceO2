#ifndef ALIHLTTPCCASETTINGS_H
#define ALIHLTTPCCASETTINGS_H

#define EXTERN_ROW_HITS
#define TRACKLET_SELECTOR_MIN_HITS(QPT) (QPT > 10 ? 10 : (QPT > 5 ? 15 : 29)) //Minimum hits should depend on Pt, low Pt tracks can have few hits. 29 Hits default, 15 for < 200 mev, 10 for < 100 mev

#define GLOBAL_TRACKING_RANGE 45					//Number of rows from the upped/lower limit to search for global track candidates in for
#define GLOBAL_TRACKING_Y_RANGE_UPPER_LEFT 0.85		//Inner portion of y-range in slice that is not used in searching for global track candidates
#define GLOBAL_TRACKING_Y_RANGE_LOWER_LEFT 0.85
#define GLOBAL_TRACKING_Y_RANGE_UPPER_RIGHT 0.85
#define GLOBAL_TRACKING_Y_RANGE_LOWER_RIGHT 0.85
#define GLOBAL_TRACKING_MIN_ROWS 10					//Min num of rows an additional global track must span over
#define GLOBAL_TRACKING_MIN_HITS 8					//Min num of hits for an additional global track
#ifdef HLTCA_STANDALONE
#define GLOBAL_TRACKING_MAINTAIN_TRACKLETS			//Maintain tracklets for standalone OpenGL event display
#endif

#define HLTCA_Y_FACTOR 4							//Weight of y residual vs z residual in tracklet constructor
#define HLTCA_GPU_MAXN 40							//Maximum number of neighbor hits to consider in one row in neightbors finder
#define TRACKLET_CONSTRUCTOR_MAX_ROW_GAP 4			//Maximum number of consecutive rows without hit in track following
#define TRACKLET_CONSTRUCTOR_MAX_ROW_GAP_SEED 2		//Same, but during fit of seed
#define MIN_TRACK_PT_DEFAULT 0.015					//Default setting for minimum track Pt at some places

#define MAX_SLICE_NTRACK (2 << 24)					//Maximum number of tracks per slice (limited by track id format)

#define HLTCA_TIMING_SUM 1

#define HLTCA_MAX_SIN_PHI_LOW 0.99f						//Must be preprocessor define because c++ pre 11 cannot use static constexpr for initializes
#define HLTCA_MAX_SIN_PHI 0.999f

#define HLTCA_ROW_COUNT 159

#define REPRODUCIBLE_CLUSTER_SORTING

#ifdef HLTCA_BUILD_O2_LIB
typedef unsigned int calink;
typedef unsigned int cahit;
#else
typedef unsigned short calink;
typedef unsigned short cahit;
#endif

#ifdef HLTCA_GPUCODE
#define ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP 6
#define ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS 3500
#else
#define ALIHLTTPCCANEIGHBOURS_FINDER_MAX_NNEIGHUP HLTCA_GPU_MAXN
#define ALIHLTTPCCASTARTHITSFINDER_MAX_FROWSTARTHITS 200000
#endif //HLTCA_GPUCODE

//#define HLTCA_FULL_CLUSTERDATA						//Store all cluster information in the cluster data, also those not needed for tracking.

#endif
