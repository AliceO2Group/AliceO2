#if !defined(SHARED_TPC_H)
#    define  SHARED_TPC_H

#define EMPTY_SPACE 2
#define TPC_ROWS_PER_CRU 18
#define TPC_PADS_PER_ROW 138
#define TPC_PADS_PER_ROW_BUFFERED (TPC_PADS_PER_ROW+2*EMPTY_SPACE)
#define TPC_MAX_TIME 1000
#define TPC_MAX_TIME_BUFFERED (TPC_MAX_TIME+2*EMPTY_SPACE)

#endif //!defined(SHARED_TPC_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
