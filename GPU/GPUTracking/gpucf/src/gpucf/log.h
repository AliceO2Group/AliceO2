#pragma once

#include <iostream>
#include <sstream>
#include <vector>


namespace log 
{

enum class Level 
{
    Debug,
    Info,
    Error,
};

const char *levelToStr(Level);

std::ostream &operator<<(std::ostream &, Level);

enum class Format 
{
    None = -1,
    ResetAll = 0,
    Bold = 1,
    ResetBold = 21,
    Underline = 4,
    ResetUnderline = 24,
    DefaultColor = 39,
    Red = 31,
    Green = 32,
    Blue = 34,
};


class FormatterBase 
{
};

template<Format... F>
class Formatter : FormatterBase 
{

public:
    std::string str() const 
    { 
        return makeEscSeq(); 
    }

private:
    static constexpr const char *ESC_SEQ_START = "\e[";
    static constexpr const char *ESC_SEQ_END   = "m";

    std::string makeEscSeq() const 
    {
        const std::vector<Format> fmts = {F...};
        std::stringstream ss;

        ss << ESC_SEQ_START;
        for (size_t i = 0; i < fmts.size(); i++) 
        {
            ss << static_cast<int>(fmts[i]) << ((i < fmts.size()-1) ? ";" : "");
        }
        ss << ESC_SEQ_END;

        return ss.str();
    }

};

template<Format... F>
std::ostream &operator<<(std::ostream &o, const Formatter<F...> &fmt) 
{
    return o << fmt.str();
}


template<Level L, class F1, class F2>
class Logger 
{

    static_assert(std::is_base_of<FormatterBase, F1>::value, "F1 must be a formatter!");
    static_assert(std::is_base_of<FormatterBase, F2>::value, "F2 must be a formatter!");

public:
    Logger() 
    {
        *this << F1() << L << Formatter<Format::ResetAll>() << " " << F2();
    }

    ~Logger() 
    { 
        *this << Formatter<Format::ResetAll>() << std::endl; 
    }

    Logger &operator<<(std::ostream &(*f)(std::ostream &)) 
    {
        f(std::cout);
        return *this;
    }

    template<typename T>
    Logger &operator<<(const T &msg) 
    {
        std::cout << msg;
        return *this;
    }

};


class ErrorShutdown 
{ 

public:
    ~ErrorShutdown() 
    { 
        std::exit(1); 
    }

};


using Debug   = Logger<Level::Debug,
      Formatter<Format::Blue>,
      Formatter<Format::DefaultColor>>;

using Info    = Logger<Level::Info,
      Formatter<Format::DefaultColor>,
      Formatter<Format::DefaultColor>>;

using Success = Logger<Level::Info,
      Formatter<Format::Green, Format::Bold>,
      Formatter<Format::Green, Format::Bold>>;

using Error   = Logger<Level::Error,
      Formatter<Format::Red, Format::Bold>,
      Formatter<Format::DefaultColor>>; 

class Fail 
    : public ErrorShutdown
    , public Error
{
};

}; // namespace log


#define ASSERT(cond) \
    if (!(cond)) log::Fail() << __FILE__ ":" << __LINE__ \
    << ": Failed assertion " #cond

#define CATCH_CL(err) \
    ASSERT(err == CL_SUCCESS) << "\n\n Encountered OpenCL error: " \
    << PrintCLError(err)
