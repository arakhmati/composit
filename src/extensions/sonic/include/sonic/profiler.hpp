#pragma once

#include <chrono>
#include <iostream>
#include <utility>

namespace sonic {
namespace profiler {
template <char... chars>
using function_name_t = std::integer_sequence<char, chars...>;

template <typename T, T... chars>
constexpr function_name_t<chars...> operator""_function_name() {
  return {};
}

template <char... chars>
std::ostream& operator<<(std::ostream& os, function_name_t<chars...>) {
  (os << chars, ...);
  return os;
}

template <auto FunctionName, auto Function>
auto timeit(auto&&... args) {
  auto start = std::chrono::system_clock::now();
  Function(std::forward<decltype(args)>(args)...);
  auto end = std::chrono::system_clock::now();
  auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e9;
  std::cout << "Function " << FunctionName << " took " << duration << " seconds to execute" << std::endl;
}
}  // namespace profiler
}  // namespace sonic