#pragma once

#include "sonic/thread_safe_queue.hpp"

#include <array>
#include <atomic>
#include <functional>
#include <thread>

namespace sonic {

namespace thread_pool {

template <auto num_threads>
class thread_pool_t {
  std::array<std::atomic<bool>, num_threads> index_to_run_thread;
  std::array<std::thread, num_threads> index_to_thread;
  sonic::thread_safe_queue::thread_safe_queue_t<std::function<void(std::size_t)>> computation_queue;
  std::atomic<std::size_t> num_remaining_computations;

 public:
  thread_pool_t() {
    std::fill(std::begin(this->index_to_run_thread), std::end(this->index_to_run_thread), true);
    this->num_remaining_computations = 0;
    auto& computation_queue = this->computation_queue;
    auto& num_remaining_computations = this->num_remaining_computations;
    for (auto index = 0; index < num_threads; index++) {
      auto& run_thread = this->index_to_run_thread[index];
      this->index_to_thread[index] = std::thread([&computation_queue, &run_thread, &num_remaining_computations, index] {
        while (run_thread) {
          auto optional_computation = computation_queue.pop_front();
          if (optional_computation.has_value()) {
            auto computation = optional_computation.value();
            computation(index);
            num_remaining_computations--;
          } else {
            std::this_thread::yield();
          }
        }
      });
    }
  }

  ~thread_pool_t() {
    this->synchronize();

    std::fill(std::begin(this->index_to_run_thread), std::end(this->index_to_run_thread), false);
    for (auto& thread : this->index_to_thread) {
      thread.join();
    }
  }

  void push_back_computation(auto&& computation) {
    num_remaining_computations++;
    this->computation_queue.push_back(computation);
  }

  void synchronize() {
    while (this->num_remaining_computations > 0) {
    }
  }
};

}  // namespace thread_pool

}  // namespace sonic