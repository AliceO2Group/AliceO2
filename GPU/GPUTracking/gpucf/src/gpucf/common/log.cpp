#include "log.h"


using namespace gpucf::log;


namespace gpucf
{
namespace log
{

const char *levelToStr(Level lvl) 
{
    switch (lvl) 
    {
    case Level::Debug: return "[Debug]";
    case Level::Info:  return "[Info ]";
    case Level::Error: return "[Error]";
    }
    return "";
}

std::ostream &operator<<(std::ostream &o, Level lvl) 
{
    return o << levelToStr(lvl);
}


} // namespace log

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

