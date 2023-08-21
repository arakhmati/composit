#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.hpp"

#include "sonic/shape.hpp"
#include "sonic/thread_safe_queue.hpp"

#include <atomic>
#include <thread>

TEST_CASE("test empty queue") {
  using namespace sonic::thread_safe_queue;

  auto queue = thread_safe_queue_t<int>{};
  CHECK(queue.empty());
  CHECK(queue.size() == 0);
}

TEST_CASE("test push and pop element") {
  using namespace sonic::thread_safe_queue;

  int element = 123;
  auto queue = thread_safe_queue_t<decltype(element)>{};

  queue.push_back(element);
  CHECK(not queue.empty());
  CHECK(queue.size() == 1);

  auto popped_element = queue.pop_front().value();
  CHECK(queue.empty());
  CHECK(queue.size() == 0);

  CHECK(popped_element == element);
}

TEST_CASE("test push and pop 3 elements") {
  using namespace sonic::thread_safe_queue;

  using element_type_t = int;

  element_type_t element_0 = 123;
  element_type_t element_1 = 321;
  element_type_t element_2 = 456;
  auto queue = thread_safe_queue_t<element_type_t>{};

  queue.push_back(element_0);
  CHECK(not queue.empty());
  CHECK(queue.size() == 1);

  queue.push_back(element_1);
  CHECK(not queue.empty());
  CHECK(queue.size() == 2);

  queue.push_back(element_2);
  CHECK(not queue.empty());
  CHECK(queue.size() == 3);

  {
    auto popped_element = queue.pop_front().value();
    CHECK(not queue.empty());
    CHECK(queue.size() == 2);
    CHECK(popped_element == element_0);
  }

  {
    auto popped_element = queue.pop_front().value();
    CHECK(not queue.empty());
    CHECK(queue.size() == 1);
    CHECK(popped_element == element_1);
  }

  {
    auto popped_element = queue.pop_front().value();
    CHECK(queue.empty());
    CHECK(queue.size() == 0);
    CHECK(popped_element == element_2);
  }
}

TEST_CASE("test push and pop 2 std::functions") {
  using namespace sonic::thread_safe_queue;

  int state = 123;
  auto function_0 = std::function([&state] { state = 321; });
  auto function_1 = std::function([] {});
  auto queue = thread_safe_queue_t<decltype(function_0)>{};

  queue.push_back(function_0);
  queue.push_back(function_1);

  CHECK(state == 123);

  {
    auto popped_function = queue.pop_front().value();
    popped_function();
    CHECK(state == 321);
  }

  { auto popped_function = queue.pop_front().value(); }
}

TEST_CASE("test queue using 2 threads") {
  using namespace sonic::thread_safe_queue;

  auto queue = thread_safe_queue_t<std::function<void()>>{};

  constexpr auto num_index_to_thread = 2;
  std::array<std::atomic<bool>, num_index_to_thread> index_to_run_thread;
  std::fill(std::begin(index_to_run_thread), std::end(index_to_run_thread), true);
  std::array<std::thread, num_index_to_thread> index_to_thread;

  std::atomic<int> num_executed_functions = 0;
  for (auto index = 0; index < num_index_to_thread; index++) {
    auto& run_thread = index_to_run_thread[index];
    index_to_thread[index] = std::thread([&queue, &run_thread, &num_executed_functions]() {
      while (run_thread) {
        if (queue.empty()) {
          std::this_thread::yield();
        } else {
          auto popped_element = queue.pop_front().value();
          popped_element();
          num_executed_functions++;
        }
      }
    });
  }

  {
    std::atomic<int> counter = 2;
    queue.push_back(std::function([&counter] { counter--; }));
    queue.push_back(std::function([&counter] { counter--; }));
    while (counter > 0) {
    }
  }

  {
    std::atomic<int> counter = 3;
    queue.push_back(std::function([&counter] { counter--; }));
    queue.push_back(std::function([&counter] { counter--; }));
    queue.push_back(std::function([&counter] { counter--; }));
    while (counter > 0) {
    }
  }

  for (auto index = 0; index < num_index_to_thread; index++) {
    index_to_run_thread[index] = false;
  }
  for (auto index = 0; index < num_index_to_thread; index++) {
    index_to_thread[index].join();
  }

  CHECK(num_executed_functions == 5);
}