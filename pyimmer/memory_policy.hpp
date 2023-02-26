#include <immer/memory_policy.hpp>

#include <Python.h>

#include <memory>

namespace pyimmer::memory_policy {

template <class T> struct allocator_t {
  typedef T value_type;

  allocator_t() = default;

  template <class U> constexpr allocator_t(const allocator_t<U> &) noexcept {}

  [[nodiscard]] static T *allocate(std::size_t size) {
    if (auto pointer =
            static_cast<void *>(PyMem_Malloc(size * sizeof(void *)))) {
      return pointer;
    }
    throw std::bad_alloc();
  }

  static void deallocate(std::size_t size, T *pointer) noexcept {
    PyMem_Free(pointer);
  }
};

using py_heap_t = allocator_t<void>;

using memory_policy_t =
    immer::memory_policy<immer::heap_policy<py_heap_t>,
                         immer::no_refcount_policy, immer::default_lock_policy,
                         immer::gc_transience_policy>;

} // namespace pyimmer::memory_policy