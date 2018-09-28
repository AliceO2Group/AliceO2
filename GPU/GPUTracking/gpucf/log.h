#pragma once

#include <ostream>
#include <sstream>


#define ESC_SEQ_START "\e["
#define ESC_SEQ_END "m"



namespace log {

enum class Level {
    Debug,
    Info,
    Error,
};

const char *levelToStr(const Level &lvl) {
    switch (lvl) {
    case Level::Debug: return "[Debug]";
    case Level::Info:  return "[Info]";
    case Level::Error: return "[Error]";
    }
}

std::ostream &operator<<(std::ostream &o, const Level &lvl) {
    return o << levelToStr(lvl);
}

enum class Format {
    None = -1,
    ResetAll = 0,
    Bold = 1,
    ResetBold = 21,
    Underline = 4,
    ResetUnderline = 24,
    DefaultColor = 39,
    Red = 31,
    Green = 32,
};


template<Format... F>
class Formatter {
    
public:
    std::string str() const {
        return makeEscSeq();
    }
    
private:
    std::string makeEscSeq() const {
        std::stringstream ss;
        ss << ESC_SEQ_START;
        for (fmt in {F...}) {
            ss << static_cast<int>(fmt);
        }
        ss << ESC_SEQ_END;
        return ss.str();
    }
    
};

template<Format... F>
std::ostream &operator<<(std::ostream &o, const Formatter<F...> &fmt) {
    return o << fmt.str();
}


template<Level L, Format... F>
class Logger {

public:
    Logger() {
        *this << Formatter<F...>() << L;      
    }
    
    ~Logger() {
        *this << Formatter<Format::ResetAll>() << std::endl;
    }
    
    template<typename T>
    Logger &operator<<(const T &msg) {
        std::cout << msg;
        return *this;
    }

};


using Debug   = Logger<Level::Debug, Format::DefaultColor>;
using Info    = Logger<Level::Info, Format::DefaultColor>;
using Success = Logger<Level::Info, Format::Green, Format::Bold>;

class Fatal : Logger<Level::Error, Format::Red, Format::Bold> {

public:
    ~Fatal() {
        std::exit(1);
    }
    
};


}; // namespace log


#define CATCH(glCall) do { \
        int err = glCall; \
        if (err) { \
            log::Fatal() << __FILE__ << ":" << __LINE__ << ": Error!"; \
        } \
    } while (false)