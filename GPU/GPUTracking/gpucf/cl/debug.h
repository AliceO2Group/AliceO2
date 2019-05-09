#if !defined(DEBUG_H)
#    define  DEBUG_H

#if defined(NDEBUG)
#  define SOFT_ASSERT(test)              ((void) 0)  
#  define DBGPR_1(str, arg1)             ((void) 0)  
#  define DBGPR_2(str, arg1, arg2)       ((void) 0)  
#  define DBGPR_3(str, arg1, arg2, arg3) ((void) 0)  
#else
#  define SOFT_ASSERT(test) \
    if (!(test)) printf("%s:%d: Failed assertion " #test "\n", __FILE__, __LINE__)

#  define DBGPR_1(str, arg1)             printf(str "\n", arg1)
#  define DBGPR_2(str, arg1, arg2)       printf(str "\n", arg1, arg2)
#  define DBGPR_3(str, arg1, arg2, arg3) printf(str "\n", arg1, arg2, arg3)
#endif

#endif // !defined(DEBUG_H)
// vim: set ts=4 sw=4 sts=4 expandtab:
