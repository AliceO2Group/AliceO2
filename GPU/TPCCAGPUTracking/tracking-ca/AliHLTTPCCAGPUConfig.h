#ifndef ALIHLTTPCCAGPUCONFIG_H
#define ALIHLTTPCCAGPUCONFIG_H

//GPU Run Configuration

#ifdef RADEON
#define HLTCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 2
#define HLTCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER 3
#define HLTCA_GPU_THREAD_COUNT 256
#define HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR 256
#define HLTCA_GPU_THREAD_COUNT_SELECTOR 256
#define HLTCA_GPU_THREAD_COUNT_FINDER 256
#define HLTCA_GPU_NUM_STREAMS 8
#elif defined(PASCAL)
#define HLTCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 2
#define HLTCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER 4
#define HLTCA_GPU_THREAD_COUNT 256
#define HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR 1024
#define HLTCA_GPU_THREAD_COUNT_SELECTOR 512
#define HLTCA_GPU_THREAD_COUNT_FINDER 512
#define HLTCA_GPU_NUM_STREAMS 8
#define HLTCA_GPU_CONSTRUCTOR_SINGLE_SLICE
//#define HLTCA_GPU_USE_TEXTURES
#elif defined(KEPLER)
#define HLTCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 4
#define HLTCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER 3
#define HLTCA_GPU_THREAD_COUNT 256
#define HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR 512
#define HLTCA_GPU_THREAD_COUNT_SELECTOR 256
#define HLTCA_GPU_THREAD_COUNT_FINDER 256
#define HLTCA_GPU_NUM_STREAMS 0
#elif defined(FERMI) || defined(__OPENCL__)
#define HLTCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 2
#define HLTCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER 3
#define HLTCA_GPU_THREAD_COUNT 256
#define HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR 256
#define HLTCA_GPU_THREAD_COUNT_SELECTOR 256
#define HLTCA_GPU_THREAD_COUNT_FINDER 256
#define HLTCA_GPU_NUM_STREAMS 0
#elif defined(TESLA)
#define HLTCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER 1
#define HLTCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER 1
#define HLTCA_GPU_THREAD_COUNT 256
#define HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR 256
#define HLTCA_GPU_THREAD_COUNT_SELECTOR 256
#define HLTCA_GPU_THREAD_COUNT_FINDER 256
#define HLTCA_GPU_NUM_STREAMS 0
#define HLTCA_GPU_USE_TEXTURES
#else
#define HLTCA_GPU_BLOCK_COUNT_CONSTRUCTOR_MULTIPLIER ??error
#define HLTCA_GPU_BLOCK_COUNT_SELECTOR_MULTIPLIER ??error
#define HLTCA_GPU_THREAD_COUNT ??error
#define HLTCA_GPU_THREAD_COUNT_CONSTRUCTOR ??error
#define HLTCA_GPU_THREAD_COUNT_SELECTOR ??error
#define HLTCA_GPU_THREAD_COUNT_FINDER ??error
#define HLTCA_GPU_NUM_STREAMS ??error
#endif

#define HLTCA_GPU_DEFAULT_HELPER_THREADS 2				//Number of helper threads to speed up initialization/output

//GPU Parameters
#define HLTCA_GPU_WARP_SIZE 32
#define HLTCA_GPU_REGS 64

#ifdef HLTCA_STANDALONE
#define HLTCA_GPU_MERGER								//Use GPU Merger
#endif

#define HLTCA_GPU_ROWALIGNMENT uint4					//Align Row Hits and Grid
#define HLTCA_GPU_ROWCOPY int							//must not be bigger than row alignment!!!

#ifdef HLTCA_GPU_USE_TEXTURES
#define HLTCA_GPU_TEXTURE_FETCH_CONSTRUCTOR				//Fetch data through texture cache
#define HLTCA_GPU_TEXTURE_FETCH_NEIGHBORS				//Fetch also in Neighbours Finder
#endif

//#define HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE		//Output Profiling Data for Tracklet Constructor Tracklet Scheduling
//#define HLTCA_GPU_TIME_PROFILE						//Output Time Profiling Data for asynchronous DMA transfer
#define BITWISE_COMPATIBLE_DEBUG_OUTPUT					//Make Debug Output of CPU and GPU bitwise compatible for comparison, also enable SORT_DUMPDATA!
#define HLTCA_GPU_SORT_DUMPDATA							//Sort Start Hits etc before dumping to file

#define HLTCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE 12
#define HLTCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT 8		//Currently must be smaller than avaiable MultiProcessors on GPU or will result in wrong results

#define HLTCA_GPU_MAX_TRACKLETS 32768					//Max Number of Tracklets that can be processed by GPU Tracker, Should be divisible by 16 at least
#define HLTCA_GPU_MAX_TRACKS 8192						//Max number of Tracks that can be processd by GPU Tracker per sector, must be below 2^24 for track ID format!!!

#define HLTCA_GPU_TRACKER_CONSTANT_MEM 65000			//Amount of Constant Memory to reserve

#define HLTCA_GPU_TRACKER_OBJECT_MEMORY		((size_t)       1024 * 1024)		//Total amount of Memory to reserve for GPU Tracker Objects
#define HLTCA_GPU_ROWS_MEMORY				((size_t)       1024 * 1024)		//Total amount of Memory to reserve for GPU Row Parameters
#define HLTCA_GPU_COMMON_MEMORY				((size_t)       1024 * 1024)		//Total amount of Memory to reserve for CommomMemoryStruct on GPU
#define HLTCA_GPU_SLICE_DATA_MEMORY			((size_t)  15 * 1024 * 1024)		//Amount of Slice Data Memory to reserve per Slice on GPU
#define HLTCA_GPU_GLOBAL_MEMORY				((size_t)  45 * 1024 * 1024)		//Amount of global temporary Memory to reserve per Slice on GPU
#define HLTCA_GPU_TRACKS_MEMORY				((size_t)  10 * 1024 * 1024)		//Amount of Memory to reserve for Final Tracks per Slice on Host as Page Locked Memory
#define HLTCA_GPU_MERGER_MEMORY				((size_t) 200 * 1024 * 1024)		//Memory for track merger
#define HLTCA_GPU_MEMALIGN					((size_t)       1024 * 1024)		//Alignment of memory blocks, all constants above must be multiple of this!!!
#define HLTCA_GPU_MEMALIGN_SMALL			((size_t)         64 * 1024)		//Alignment of small blocks, HLTCA_GPU_MEMALIGN must be multiple of this!!!

//Make sure options do not interfere

#ifndef HLTCA_GPUCODE
//No texture fetch for CPU Tracker
#ifdef HLTCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
#undef HLTCA_GPU_TEXTURE_FETCH_CONSTRUCTOR
#endif
#ifdef HLTCA_GPU_TEXTURE_FETCH_NEIGHBORS
#undef HLTCA_GPU_TEXTURE_FETCH_NEIGHBORS
#endif

//Do not cache Row Hits during Tracklet selection in Registers for CPU Tracker
#undef HLTCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE
#define HLTCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE 0
#else
//Sort start hits for GPU tracker
#define HLTCA_GPU_SORT_STARTHITS
#endif

//Error Codes for GPU Tracker
#define HLTCA_GPU_ERROR_NONE 0
#define HLTCA_GPU_ERROR_ROWBLOCK_TRACKLET_OVERFLOW 1
#define HLTCA_GPU_ERROR_TRACKLET_OVERFLOW 2
#define HLTCA_GPU_ERROR_TRACK_OVERFLOW 3
#define HLTCA_GPU_ERROR_SCHEDULE_COLLISION 4
#define HLTCA_GPU_ERROR_WRONG_ROW 5
#define HLTCA_GPU_ERROR_STARTHIT_OVERFLOW 6
#define HLTCA_GPU_ERROR_STRINGS {"HLTCA_GPU_ERROR_NONE", "HLTCA_GPU_ERROR_ROWBLOCK_TRACKLET_OVERFLOW", "HLTCA_GPU_ERROR_TRACKLET_OVERFLOW", "HLTCA_GPU_ERROR_TRACK_OVERFLOW", "HLTCA_GPU_ERROR_SCHEDULE_COLLISION", "HLTCA_GPU_ERROR_WRONG_ROW", "HLTCA_GPU_ERROR_STARTHIT_OVERFLOW"}

#endif
