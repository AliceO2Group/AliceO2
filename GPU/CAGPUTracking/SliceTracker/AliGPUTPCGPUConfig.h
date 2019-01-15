#ifndef ALIHLTTPCCAGPUCONFIG_H
#define ALIHLTTPCCAGPUCONFIG_H

//GPU Run Configuration

#ifdef GPUCA_GPUTYPE_RADEON
#define GPUCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 2
#define GPUCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER 3
#define GPUCA_GPU_THREAD_COUNT 256
#define GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR 256
#define GPUCA_GPU_THREAD_COUNT_SELECTOR 256
#define GPUCA_GPU_THREAD_COUNT_FINDER 256
#define GPUCA_GPU_NUM_STREAMS 8
#elif defined(GPUCA_GPUTYPE_PASCAL)
#define GPUCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 2
#define GPUCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER 4
#define GPUCA_GPU_THREAD_COUNT 256
#define GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR 1024
#define GPUCA_GPU_THREAD_COUNT_SELECTOR 512
#define GPUCA_GPU_THREAD_COUNT_FINDER 512
#define GPUCA_GPU_NUM_STREAMS 8
#define GPUCA_GPU_CONSTRUCTOR_SINGLE_SLICE
//#define GPUCA_GPU_USE_TEXTURES
#elif defined(GPUCA_GPUTYPE_KEPLER)
#define GPUCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 4
#define GPUCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER 3
#define GPUCA_GPU_THREAD_COUNT 256
#define GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR 512
#define GPUCA_GPU_THREAD_COUNT_SELECTOR 256
#define GPUCA_GPU_THREAD_COUNT_FINDER 256
#define GPUCA_GPU_NUM_STREAMS 0
#elif defined(GPUCA_GPUTYPE_FERMI) || defined(__OPENCL__)
#define GPUCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 2
#define GPUCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER 3
#define GPUCA_GPU_THREAD_COUNT 256
#define GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR 256
#define GPUCA_GPU_THREAD_COUNT_SELECTOR 256
#define GPUCA_GPU_THREAD_COUNT_FINDER 256
#define GPUCA_GPU_NUM_STREAMS 0
#elif defined(GPUCA_GPUTYPE_TESLA)
#define GPUCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 1
#define GPUCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER 1
#define GPUCA_GPU_THREAD_COUNT 256
#define GPUCA_GPU_THREAD_COUNT_CONSTRUCTOR 256
#define GPUCA_GPU_THREAD_COUNT_SELECTOR 256
#define GPUCA_GPU_THREAD_COUNT_FINDER 256
#define GPUCA_GPU_NUM_STREAMS 0
#define GPUCA_GPU_USE_TEXTURES
#elif defined(GPUCA_GPUCODE)
#error GPU TYPE NOT SET
#endif

#define GPUCA_GPU_DEFAULT_HELPER_THREADS 2				//Number of helper threads to speed up initialization/output

#ifdef GPUCA_STANDALONE
#define GPUCA_GPU_MERGER								//Use GPU Merger
#endif

#define GPUCA_GPU_ROWALIGNMENT uint4					//Align Row Hits and Grid

#ifdef GPUCA_GPU_USE_TEXTURES
#define GPUCA_GPU_TEXTURE_FETCH_CONSTRUCTOR				//Fetch data through texture cache
#define GPUCA_GPU_TEXTURE_FETCH_NEIGHBORS				//Fetch also in Neighbours Finder
#endif

//#define GPUCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE		//Output Profiling Data for Tracklet Constructor Tracklet Scheduling
//#define GPUCA_GPU_TIME_PROFILE						//Output Time Profiling Data for asynchronous DMA transfer

#define GPUCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE 12
#define GPUCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT 8		//Currently must be smaller than avaiable MultiProcessors on GPU or will result in wrong results

#define GPUCA_GPU_MAX_TRACKLETS 65536					//Max Number of Tracklets that can be processed by GPU Tracker, Should be divisible by 16 at least
#define GPUCA_GPU_MAX_TRACKS 32768						//Max number of Tracks that can be processd by GPU Tracker per sector, must be below 2^24 for track ID format!!!
#define GPUCA_GPU_MAX_ROWSTARTHITS 20000				//Maximum number of start hits per row

#define GPUCA_GPU_TRACKER_CONSTANT_MEM 65000			//Amount of Constant Memory to reserve

#define GPUCA_GPU_TRACKER_OBJECT_MEMORY		((size_t)       1024 * 1024)		//Total amount of Memory to reserve for GPU Tracker Objects
#define GPUCA_GPU_ROWS_MEMORY				((size_t)       1024 * 1024)		//Total amount of Memory to reserve for GPU Row Parameters
#define GPUCA_GPU_COMMON_MEMORY				((size_t)       1024 * 1024)		//Total amount of Memory to reserve for CommomMemoryStruct on GPU
#define GPUCA_GPU_SLICE_DATA_MEMORY			((size_t)  50 * 1024 * 1024)		//Amount of Slice Data Memory to reserve per Slice on GPU
#define GPUCA_GPU_GLOBAL_MEMORY				((size_t) 100 * 1024 * 1024)		//Amount of global temporary Memory to reserve per Slice on GPU
#define GPUCA_GPU_TRACKS_MEMORY				((size_t)  10 * 1024 * 1024)		//Amount of Memory to reserve for Final Tracks per Slice on Host as Page Locked Memory
#define GPUCA_GPU_MERGER_MEMORY				((size_t) 600 * 1024 * 1024)		//Memory for track merger
#define GPUCA_GPU_MEMALIGN					((size_t)       1024 * 1024)		//Alignment of memory blocks, all constants above must be multiple of this!!!
#define GPUCA_GPU_MEMALIGN_SMALL			((size_t)         64 * 1024)		//Alignment of small blocks, GPUCA_GPU_MEMALIGN must be multiple of this!!!
#define GPUCA_GPU_MEMORY_SIZE				((size_t) 500 * 1024 * 1024)		//Size of memory allocated on Device
#define GPUCA_HOST_MEMORY_SIZE				((size_t) 500 * 1024 * 1024)		//Size of memory allocated on Host

//Make sure options do not interfere

#ifndef GPUCA_GPUCODE
//No texture fetch for CPU Tracker
#ifdef GPUCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
#undef GPUCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
#endif
#ifdef GPUCA_GPU_TEXTURE_FETCH_NEIGHBORS
#undef GPUCA_GPU_TEXTURE_FETCH_NEIGHBORS
#endif

//Do not cache Row Hits during Tracklet selection in Registers for CPU Tracker
#undef GPUCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE
#define GPUCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE 0
#else
//Sort start hits for GPU tracker
#define GPUCA_GPU_SORT_STARTHITS
#endif

//Error Codes for GPU Tracker
#define GPUCA_GPU_ERROR_NONE 0
#define GPUCA_GPU_ERROR_ROWBLOCK_TRACKLET_OVERFLOW 1
#define GPUCA_GPU_ERROR_TRACKLET_OVERFLOW 2
#define GPUCA_GPU_ERROR_TRACK_OVERFLOW 3
#define GPUCA_GPU_ERROR_SCHEDULE_COLLISION 4
#define GPUCA_GPU_ERROR_WRONG_ROW 5
#define GPUCA_GPU_ERROR_STARTHIT_OVERFLOW 6
#define GPUCA_GPU_ERROR_STRINGS {"GPUCA_GPU_ERROR_NONE", "GPUCA_GPU_ERROR_ROWBLOCK_TRACKLET_OVERFLOW", "GPUCA_GPU_ERROR_TRACKLET_OVERFLOW", "GPUCA_GPU_ERROR_TRACK_OVERFLOW", "GPUCA_GPU_ERROR_SCHEDULE_COLLISION", "GPUCA_GPU_ERROR_WRONG_ROW", "GPUCA_GPU_ERROR_STARTHIT_OVERFLOW"}

#endif
