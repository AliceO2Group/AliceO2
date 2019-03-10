#include "Fragment.h"


using namespace gpucf;


Fragment::Fragment(size_t items)
    : Fragment(0, 0, items, 0)
{
}

Fragment::Fragment(size_t start, size_t backlog, size_t items, size_t future)
    : start(start)
    , backlog(backlog)
    , items(items)
    , future(future)
{
}

// vim: set ts=4 sw=4 sts=4 expandtab:
