#pragma once

#include <CL/cl2.hpp>

namespace sonic {
namespace opencl {

cl::Device get_default_device() {
  cl::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  if (platforms.empty()) {
#ifdef _GLIBCXX_IOSTREAM
    std::cerr << "No platforms found!" << std::endl;
#endif
    exit(1);
  }

  auto platform = platforms.front();
  cl::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

  if (devices.empty()) {
#ifdef _GLIBCXX_IOSTREAM
    std::cerr << "No devices found!" << std::endl;
#endif
    exit(1);
  }
  return devices.front();
}

cl::Program create_program(cl::Context context, const char* kernel_source) {
  cl::Program::Sources sources{kernel_source};
  cl::Program program = cl::Program(context, sources);

  auto build_status = program.build();
  if (build_status != CL_BUILD_SUCCESS) {
    auto device = context.getInfo<CL_CONTEXT_DEVICES>().at(0);
#ifdef _GLIBCXX_IOSTREAM
    std::cerr << "Error!\nBuild Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) << "\nBuild Log:\t "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
#endif
    exit(1);
  }
  return program;
}

}  // namespace opencl
}  // namespace sonic