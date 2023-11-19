#pragma once

#include <mutex>
#include <optional>
#include <queue>

namespace sonic {

namespace thread_safe_queue {

template <class T>
class thread_safe_queue_t {
  std::queue<T> queue;
  mutable std::mutex mutex;

 public:
  ~thread_safe_queue_t() {}

  void push_back(T value) {
    std::scoped_lock<std::mutex> lock(mutex);
    this->queue.push(value);
  }

  std::optional<T> pop_front() {
    std::unique_lock<std::mutex> lock(mutex);
    if (this->queue.empty()) {
      return std::nullopt;
    }
    auto value = this->queue.front();
    this->queue.pop();
    return value;
  }

  bool empty() const {
    std::scoped_lock<std::mutex> lock(mutex);
    return this->queue.empty();
  }

  std::size_t size() const {
    std::scoped_lock<std::mutex> lock(mutex);
    return this->queue.size();
  }
};

}  // namespace thread_safe_queue

}  // namespace sonic