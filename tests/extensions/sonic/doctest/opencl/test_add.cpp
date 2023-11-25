#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.hpp"

#include <array>
#include <iostream>

#include <CL/cl2.hpp>

#include "sonic/lazy_computation.hpp"
#include "sonic/opencl.hpp"
#include "sonic/profiler.hpp"

template <typename data_type_t, std::size_t size>
bool assert_equal(const data_type_t* tensor_a, const data_type_t* tensor_b) {
  for (std::size_t i = 0; i < size; i++) {
    if (tensor_a[i] != tensor_b[i]) {
      std::cout << tensor_a[i] << " != " << tensor_b[i] << std::endl;
      return false;
    }
  }
  return true;
}

template <typename data_type_t, std::size_t size>
void add(const sonic::tensor::aligned_array<data_type_t, size>& input_buffer_a,
         const sonic::tensor::aligned_array<data_type_t, size>& input_buffer_b,
         sonic::tensor::aligned_array<data_type_t, size>& output_buffer) {
  using namespace sonic::lazy_computation;
  using namespace sonic::tensor;

  const auto input_a = as_lazy_computation<data_type_t, sonic::shape::shape_t<size>>(input_buffer_a);
  const auto input_b = as_lazy_computation<data_type_t, sonic::shape::shape_t<size>>(input_buffer_b);
  const auto output = input_a + input_b;
  evaluate_to<vector8_float32>(output, output_buffer.data());
}

template <typename data_type_t, std::size_t size>
void opencl_add(cl::CommandQueue queue,
                const cl::Program& program,
                cl::Buffer cl_input_buffer_a,
                cl::Buffer cl_input_buffer_b,
                cl::Buffer cl_output_buffer) {
  cl::Kernel kernel(program, "add");
  kernel.setArg(0, cl_input_buffer_a);
  kernel.setArg(1, cl_input_buffer_b);
  kernel.setArg(2, cl_output_buffer);

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(size));
  queue.finish();
}

namespace detail {
template <class>
inline constexpr bool throw_unsupported_data_type_t = false;
}

template <typename data_type_t>
constexpr auto get_kernel_source() {
  if constexpr (std::is_same_v<data_type_t, std::int32_t>) {
    return R"(
      __kernel void
      add(const global int* input_a, const global int* input_b, global int* output) {
        int index = get_global_id(0);
        output[index] = input_a[index] + input_b[index];
      }
    )";
  } else if constexpr (std::is_same_v<data_type_t, float>) {
    return R"(
      __kernel void
      add(const global float* input_a, const global float* input_b, global float* output) {
        int index = get_global_id(0);
        output[index] = input_a[index] + input_b[index];
      }
    )";
  } else {
    static_assert(detail::throw_unsupported_data_type_t<data_type_t>);
  }
}

TEST_CASE("test opencl add") {
  using namespace sonic::opencl;
  using namespace sonic::profiler;
  using namespace sonic::tensor;

  using data_type_t = float;
  constexpr std::size_t size = 1 << 20;

  aligned_array<data_type_t, size> input_buffer_a{};
  input_buffer_a.fill(3);

  aligned_array<data_type_t, size> input_buffer_b{};
  input_buffer_b.fill(5);

  aligned_array<data_type_t, size> golden_output_buffer{};
  aligned_array<data_type_t, size> actual_output_buffer{};

  timeit<"add"_function_name, add<data_type_t, size>>(input_buffer_a, input_buffer_b, golden_output_buffer);

  const auto device = get_default_device();
  const auto context = cl::Context(device);
  const auto program = create_program(context, get_kernel_source<data_type_t>());
  auto queue = cl::CommandQueue(context, device);

  cl::Buffer cl_input_buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                               size * sizeof(data_type_t), input_buffer_a.data());
  cl::Buffer cl_input_buffer_b(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                               size * sizeof(data_type_t), input_buffer_b.data());
  cl::Buffer cl_output_buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, size * sizeof(data_type_t));
  timeit<"opencl_add"_function_name, opencl_add<data_type_t, size>>(queue, program, cl_input_buffer_a,
                                                                    cl_input_buffer_b, cl_output_buffer);
  queue.enqueueReadBuffer(cl_output_buffer, CL_TRUE, 0, size * sizeof(data_type_t), actual_output_buffer.data());

  bool equal = assert_equal<data_type_t, size>(golden_output_buffer.data(), actual_output_buffer.data());
  CHECK(equal);
}