#if !defined(SHARED_TPC_H)
#    define  SHARED_TPC_H

#define PADDING 2
#define TPC_ROWS_PER_CRU 18
#define TPC_PADS_PER_ROW 138
#define TPC_PADS_PER_ROW_PADDED (TPC_PADS_PER_ROW+2*PADDING)
#define TPC_MAX_TIME 1000
#define TPC_MAX_TIME_PADDED (TPC_MAX_TIME+2*PADDING)

#endif //!defined(SHARED_TPC_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
