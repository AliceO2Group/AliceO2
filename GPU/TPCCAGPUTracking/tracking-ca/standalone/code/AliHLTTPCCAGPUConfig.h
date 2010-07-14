#ifndef ALIHLTTPCCAGPUCONFIG_H
#define ALIHLTTPCCAGPUCONFIG_H

//GPU Run Configuration
#define HLTCA_GPU_BLOCK_COUNT 30
#define HLTCA_GPU_THREAD_COUNT 256

//GPU Parameters
#define HLTCA_GPU_WARP_SIZE 32
#define HLTCA_GPU_REGS 64

//Detector Parameters
#define HLTCA_ROW_COUNT 159

#define HLTCA_GPU_ROWALIGNMENT uint4					//Align Row Hits and Grid
#define HLTCA_GPU_ROWCOPY int							//must not be bigger than row alignment!!!

#define HLTCA_GPU_SCHED_ROW_STEP 32						//Amount of Rows to process in one step before rescheduling
#define HLTCA_GPU_SCHED_FIXED_START						//Assign each GPU thread a start tracklet to start with instead of using the scheduler to obtain start tracklet
//#define HLTCA_GPU_SCHED_FIXED_SLICE					//Make each Multiprocessor on GPU work only on a single slice during tracklet construction
#define HLTCA_GPU_RESCHED								//Use dynamic tracklet scheduling

#define HLTCA_GPU_TEXTURE_FETCH							//Fetch data through texture cache
#define HLTCA_GPU_TEXTURE_FETCHa						//Fetch also in Neighbours Finder

//#define HLTCA_GPU_TRACKLET_CONSTRUCTOR_DO_PROFILE		//Output Profiling Data for Tracklet Constructor Tracklet Scheduling
//#define HLTCA_GPU_TIME_PROFILE						//Output Time Profiling Data for asynchronous DMA transfer

#define HLTCA_GPU_TRACKLET_SELECTOR_HITS_REG_SIZE 12
#define HLTCA_GPU_TRACKLET_SELECTOR_SLICE_COUNT 3		//Currently must be smaller than avaiable MultiProcessors on GPU or will result in wrong results

#define HLTCA_GPU_SORT_DUMPDATA							//Sort Start Hits etc before dumping to file

#define HLTCA_GPU_MAX_TRACKLETS 12288					//Max Number of Tracklets that can be processed by GPU Tracker, Should be divisible by 16 at least
#define HLTCA_GPU_MAX_TRACKS 3072						//Max number of Tracks that can be processd by GPU Tracker

//#define HLTCA_GPU_EMULATION_SINGLE_TRACKLET 1313		//Run Tracklet constructor on on single Tracklet in Device Emulation Mode
//#define HLTCA_GPU_EMULATION_DEBUG_TRACKLET 1313

#define HLTCA_GPU_TRACKER_CONSTANT_MEM 65000			//Amount of Constant Memory to reserve

#define HLTCA_GPU_TRACKER_OBJECT_MEMORY 1024 * 1024		//Total amount of Memory to reserve for GPU Tracker Objects
#define HLTCA_GPU_ROWS_MEMORY 1024 * 1024				//Total amount of Memory to reserve for GPU Row Parameters
#define HLTCA_GPU_COMMON_MEMORY 1024 * 1024				//Total amount of Memory to reserve for CommomMemoryStruct on GPU
#define HLTCA_GPU_SLICE_DATA_MEMORY 7 * 1024 * 1024		//Amount of Slice Data Memory to reserve per Slice on GPU
#define HLTCA_GPU_GLOBAL_MEMORY 16 * 1024 * 1024		//Amount of global temporary Memory to reserve per Slice on GPU
#define HLTCA_GPU_TRACKS_MEMORY 2 * 1024 * 1024			//Amount of Memory to reserve for Final Tracks per Slice on Host as Page Locked Memory

//Make sure options do not interfere

#ifndef HLTCA_GPUCODE
//No texture fetch for CPU Tracker
#ifdef HLTCA_GPU_TEXTURE_FETCH
#undef HLTCA_GPU_TEXTURE_FETCH
#endif
#ifdef HLTCA_GPU_TEXTURE_FETCHa
#undef HLTCA_GPU_TEXTURE_FETCHa
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

#endif

