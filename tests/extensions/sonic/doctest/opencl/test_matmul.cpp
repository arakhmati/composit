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

template <typename data_type_t, std::size_t m_size, std::size_t k_size, std::size_t n_size>
void matmul(const sonic::tensor::aligned_array<data_type_t, m_size * k_size>& input_buffer_a,
            const sonic::tensor::aligned_array<data_type_t, k_size * n_size>& input_buffer_b,
            sonic::tensor::aligned_array<data_type_t, m_size * n_size>& output_buffer) {
  using namespace sonic::lazy_computation;
  using namespace sonic::tensor;

  const auto input_a = as_lazy_computation<data_type_t, sonic::shape::shape_t<1, m_size, k_size>>(input_buffer_a);
  const auto input_b = as_lazy_computation<data_type_t, sonic::shape::shape_t<k_size, n_size>>(input_buffer_b);
  matmul(input_a, input_b, output_buffer.data())();
}

template <typename data_type_t, std::size_t m_size, std::size_t k_size, std::size_t n_size>
void opencl_matmul(cl::CommandQueue queue,
                   const cl::Program& program,
                   cl::Buffer cl_input_buffer_a,
                   cl::Buffer cl_input_buffer_b,
                   cl::Buffer cl_output_buffer) {
  cl::Kernel kernel(program, "matmul");
  kernel.setArg(0, cl_input_buffer_a);
  kernel.setArg(1, cl_input_buffer_b);
  kernel.setArg(2, cl_output_buffer);
  const auto m = m_size;
  kernel.setArg(3, sizeof(unsigned int), &m);
  const auto k = k_size;
  kernel.setArg(4, sizeof(unsigned int), &k);
  const auto n = n_size;
  kernel.setArg(5, sizeof(unsigned int), &n);

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(m_size, n_size), cl::NDRange(16, 16));
  queue.finish();
}

namespace detail {
template <class>
inline constexpr bool throw_unsupported_data_type_t = false;
}

const char* kernel_source = R"(
  __kernel void
  matmul(
    const global float* input_a,
    const global float* input_b,
    global float* output,
    const int m_size,
    const int k_size,
    const int n_size) {

      const int tile_size = 16;

      int tile_height_index = get_local_id(0);
      int tile_width_index = get_local_id(1);
      int m_index = get_global_id(0);
      int n_index = get_global_id(1);
      int index = (m_index * n_size) + n_index;

      __local float tile_a[tile_size][tile_size];
      __local float tile_b[tile_size][tile_size];

      float accumulator = 0;

      const int num_k_tiles = k_size / tile_size;
      for(int k_index = 0; k_index < num_k_tiles; k_index++){
          const int height_index = k_index * tile_size + tile_height_index;
          const int width_index = k_index * tile_size + tile_width_index;
          tile_a[tile_height_index][tile_width_index] = input_a[m_index * k_size + width_index];
          tile_b[tile_height_index][tile_width_index] = input_b[height_index * n_size + n_index];

          barrier(CLK_LOCAL_MEM_FENCE);

          for(int k = 0; k < tile_size; k++){
              accumulator += tile_a[tile_height_index][k] * tile_b[k][tile_width_index];
          }

          barrier(CLK_LOCAL_MEM_FENCE);
      }
      output[index] = accumulator;
  }
    )";

TEST_CASE("test opencl matmul") {
  using namespace sonic::opencl;
  using namespace sonic::profiler;
  using namespace sonic::tensor;

  using data_type_t = float;
  constexpr std::size_t m_size = 4096;
  constexpr std::size_t k_size = 4096;
  constexpr std::size_t n_size = 4096;

  aligned_array<data_type_t, m_size * k_size> input_buffer_a{};
  input_buffer_a.fill(3);

  aligned_array<data_type_t, k_size * n_size> input_buffer_b{};
  input_buffer_b.fill(5);

  aligned_array<data_type_t, m_size * n_size> golden_output_buffer{};
  aligned_array<data_type_t, m_size * n_size> actual_output_buffer{};

  timeit<"matmul"_function_name, matmul<data_type_t, m_size, k_size, n_size>>(input_buffer_a, input_buffer_b,
                                                                              golden_output_buffer);

  const auto device = get_default_device();
  const auto context = cl::Context(device);
  const auto program = create_program(context, kernel_source);
  auto queue = cl::CommandQueue(context, device);

  cl::Buffer cl_input_buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                               m_size * k_size * sizeof(data_type_t), input_buffer_a.data());
  cl::Buffer cl_input_buffer_b(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                               k_size * n_size * sizeof(data_type_t), input_buffer_b.data());
  cl::Buffer cl_output_buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                              m_size * n_size * sizeof(data_type_t));
  timeit<"opencl_matmul"_function_name, opencl_matmul<data_type_t, m_size, k_size, n_size>>(
      queue, program, cl_input_buffer_a, cl_input_buffer_b, cl_output_buffer);

  queue.enqueueReadBuffer(cl_output_buffer, CL_TRUE, 0, m_size * n_size * sizeof(data_type_t),
                          actual_output_buffer.data());

  bool equal = assert_equal<data_type_t, m_size * n_size>(golden_output_buffer.data(), actual_output_buffer.data());
  CHECK(equal);
}