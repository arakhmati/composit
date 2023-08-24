#pragma once

#include "sonic/thread_safe_queue.hpp"

#include <array>
#include <atomic>
#include <thread>

namespace sonic {

namespace thread_pool {

template <auto num_threads>
class thread_pool_t {
  std::array<std::atomic<bool>, num_threads> index_to_run_thread;
  std::array<std::thread, num_threads> index_to_thread;
  sonic::thread_safe_queue::thread_safe_queue_t<std::function<void(std::size_t)>> work_queue;

 public:
  thread_pool_t() {
    auto& work_queue = this->work_queue;
    std::fill(std::begin(this->index_to_run_thread), std::end(this->index_to_run_thread), true);
    for (auto index = 0; index < num_threads; index++) {
      auto& run_thread = this->index_to_run_thread[index];
      this->index_to_thread[index] = std::thread([&work_queue, &run_thread, index]() {
        while (run_thread) {
          auto optional_computation = work_queue.pop_front();
          if (optional_computation.has_value()) {
            auto computation = optional_computation.value();
            computation(index);
          } else {
            std::this_thread::yield();
          }
        }
      });
    }
  }

  ~thread_pool_t() {
    while (not this->work_queue.empty()) {
    }

    std::fill(std::begin(this->index_to_run_thread), std::end(this->index_to_run_thread), false);
    for (auto& thread : this->index_to_thread) {
      thread.join();
    }
  }

  void push_back_computation(auto&& computation) { this->work_queue.push_back(computation); }
};

}  // namespace thread_pool

}  // namespace sonic