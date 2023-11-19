#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.hpp"

#include <iostream>
#include <vector>

#include <CL/cl2.hpp>

#include "sonic/opencl.hpp"
#include "sonic/profiler.hpp"

template <typename data_type_t, std::size_t Size>
bool assert_equal(const data_type_t* tensor_a, const data_type_t* tensor_b) {
  for (std::size_t i = 0; i < Size; i++) {
    if (tensor_a[i] != tensor_b[i]) {
      std::cout << tensor_a[i] << " != " << tensor_b[i] << std::endl;
      return false;
    }
  }
  return true;
}

template <typename data_type_t, std::size_t Size>
void add(const data_type_t* input_a, const data_type_t* input_b, data_type_t* output) {
  for (std::size_t i = 0; i < Size; i++) {
    output[i] = input_a[i] + input_b[i];
  }
}

template <typename data_type_t, std::size_t Size>
void opencl_add(const cl::Context& context,
                const cl::Device& device,
                const cl::Program& program,
                data_type_t* input_a,
                data_type_t* input_b,
                data_type_t* output) {
  cl::Buffer input_buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                            Size * sizeof(data_type_t), input_a);
  cl::Buffer input_buffer_b(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                            Size * sizeof(data_type_t), input_b);
  cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, Size * sizeof(data_type_t));

  cl::Kernel kernel(program, "add");
  kernel.setArg(0, input_buffer_a);
  kernel.setArg(1, input_buffer_b);
  kernel.setArg(2, output_buffer);

  cl::CommandQueue queue(context, device);
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(Size));
  queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, Size * sizeof(data_type_t), output);
}

const char* kernel_source = R"(
   __kernel void add(const __global int* input_a, const __global int* input_b, __global int* output){
       int index = get_global_id(0);
       output[index] = input_a[index] + input_b[index];
   }
)";

TEST_CASE("test opencl add") {
  using namespace sonic::opencl;
  using namespace sonic::profiler;

  constexpr std::size_t Size = 1 << 20;
  using data_type_t = std::uint32_t;

  std::array<data_type_t, Size> input_a{};
  input_a.fill(3);

  std::array<data_type_t, Size> input_b{};
  input_b.fill(5);

  std::array<data_type_t, Size> golden_output{};
  std::array<data_type_t, Size> actual_output{};

  timeit<"add"_function_name, add<data_type_t, Size>>(input_a.data(), input_b.data(), golden_output.data());

  const auto device = get_default_device();
  const auto context = cl::Context(device);
  const auto program = create_program(context, kernel_source);
  timeit<"opencl_add"_function_name, opencl_add<data_type_t, Size>>(context, device, program, input_a.data(),
                                                                    input_b.data(), actual_output.data());

  bool equal = assert_equal<data_type_t, Size>(golden_output.data(), actual_output.data());
  CHECK(equal);
}