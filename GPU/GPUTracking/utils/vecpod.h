#include <vector>

template <class T> struct vecpod_allocator {
  typedef T value_type;
  vecpod_allocator() noexcept : stdalloc() {}
  T* allocate (std::size_t n) { return stdalloc.allocate(n); }
  void deallocate (T* p, std::size_t n) { stdalloc.deallocate(p, n); }
  static void construct(T*) {}
  std::allocator<T> stdalloc;
};

template <class T> using vecpod = typename std::vector<T, vecpod_allocator<T>>;
//template <class T> using vecpod = typename std::vector<T>;
