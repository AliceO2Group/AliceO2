#ifndef DATA_DUMP_H
#define DATA_DUMP_H

#include <atomic>
#include <iostream>
#include <typeinfo>
#include <type_traits>

class DataDumpFlag
{
 public:
  DataDumpFlag(const DataDumpFlag& other) = delete;
  DataDumpFlag(DataDumpFlag&& other) noexcept = delete;

  DataDumpFlag& operator=(const DataDumpFlag& other) = delete;
  DataDumpFlag& operator=(DataDumpFlag&& other) noexcept = delete;

  static DataDumpFlag& getInstance() noexcept
  {
    static DataDumpFlag instance;
    return instance;
  }

  explicit operator bool() noexcept
  {
    return test();
  }
  bool test() noexcept
  {
    // std::atomic_flag::test available in C++20
    bool state = flag.test_and_set();
    if (not state)
      unset();
    return state;
  }

  void set() noexcept
  {
    flag.test_and_set();
  }
  void unset() noexcept
  {
    flag.clear();
  }

 private:
  DataDumpFlag() noexcept
  {
    // std::atomic_flag's default constructor sets it to clear (false) in C++20
    flag.clear();
  }
  // Unlike std::atomic<bool>, std::atomic_flag is guaranteed to be lock-free
  std::atomic_flag flag;
};

inline bool isDataDumpFlag(const std::string& flag) noexcept
{
  return flag == "--data-dump";
}

inline bool isDataDumpOff() noexcept
{
  return not DataDumpFlag::getInstance().test();
}

#define CLASS_INSPECTOR(Instance, Call)                              \
  template <typename T>                                              \
  class Instance                                                     \
  {                                                                  \
   private:                                                          \
    template <typename U>                                            \
    using Type = decltype(std::declval<U>().Call, std::true_type{}); \
                                                                     \
    template <typename U>                                            \
    static Type<U> f(std::remove_reference_t<U>*);                   \
    template <typename>                                              \
    static std::false_type f(...);                                   \
                                                                     \
   public:                                                           \
    static bool const value = decltype(f<T>(0))::value;              \
  };

template <typename T>
const char* type_name() noexcept
{
  return typeid(T).name();
}

CLASS_INSPECTOR(has_Dump, Dump());
template <typename T>
inline constexpr bool has_Dump_v = has_Dump<T>::value;

CLASS_INSPECTOR(has_Print, Print());
template <typename T>
inline constexpr bool has_Print_v = has_Print<T>::value;

CLASS_INSPECTOR(has_print, print());
template <typename T>
inline constexpr bool has_print_v = has_print<T>::value;

template <typename T>
void dumpData(T const& object)
{
  if (isDataDumpOff()) {
    return;
  }
  if constexpr (has_Dump_v<const T>) {
    object.Dump();
  } else if constexpr (has_Print_v<const T>) {
    object.Print();
  } else if constexpr (has_print_v<const T>) {
    object.print();
  } else if constexpr (std::is_arithmetic_v<T>) {
    std::cout << object << std::endl;
  } else {
    std::cout << "Unknown dump method for type " << type_name<T>() << std::endl;
  }
}

#endif // DATA_DUMP_H
