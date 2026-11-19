//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// disk_scheduler_thread_safety_test.cpp
//
// Identification: test/storage/disk_scheduler_thread_safety_test.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <future>  // NOLINT
#include <memory>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "storage/disk/disk_manager_memory.h"
#include "storage/disk/disk_scheduler.h"

namespace bustub {

using bustub::DiskManagerUnlimitedMemory;

/**
 * @brief Many producers write and read their own pages concurrently.
 *
 * This checks that `Schedule` can be called from multiple threads at the same
 * time, that every request is eventually processed by the background worker,
 * and that requests for different pages do not corrupt each other.
 */
// NOLINTNEXTLINE
TEST(DiskSchedulerThreadSafetyTest, DISABLED_ConcurrentWriteReadDistinctPagesTest) {
  constexpr int kNumThreads = 8;
  constexpr int kPagesPerThread = 8;

  auto dm = std::make_unique<DiskManagerUnlimitedMemory>();
  auto disk_scheduler = std::make_unique<DiskScheduler>(dm.get());

  std::promise<void> start_promise;
  auto start_future = start_promise.get_future().share();
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int thread_id = 0; thread_id < kNumThreads; ++thread_id) {
    threads.emplace_back([&, thread_id] {
      start_future.wait();
      for (int i = 0; i < kPagesPerThread; ++i) {
        const page_id_t page_id = thread_id * kPagesPerThread + i;

        std::array<char, BUSTUB_PAGE_SIZE> write_buf{};
        std::snprintf(write_buf.data(), write_buf.size(), "thread-%d-page-%d", thread_id, i);

        auto write_promise = disk_scheduler->CreatePromise();
        auto write_future = write_promise.get_future();
        std::vector<DiskRequest> write_requests;
        write_requests.push_back(DiskRequest{/*is_write=*/true, write_buf.data(), page_id, std::move(write_promise)});
        disk_scheduler->Schedule(write_requests);
        if (!write_future.get()) {
          failures.fetch_add(1);
        }

        std::array<char, BUSTUB_PAGE_SIZE> read_buf{};
        auto read_promise = disk_scheduler->CreatePromise();
        auto read_future = read_promise.get_future();
        std::vector<DiskRequest> read_requests;
        read_requests.push_back(DiskRequest{/*is_write=*/false, read_buf.data(), page_id, std::move(read_promise)});
        disk_scheduler->Schedule(read_requests);
        if (!read_future.get()) {
          failures.fetch_add(1);
        }

        if (std::memcmp(write_buf.data(), read_buf.data(), BUSTUB_PAGE_SIZE) != 0) {
          failures.fetch_add(1);
        }
      }
    });
  }

  // Release all producer threads at roughly the same time to maximize contention.
  start_promise.set_value();
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(failures.load(), 0);

  disk_scheduler.reset();
  dm->ShutDown();
}

/**
 * @brief Concurrent writers and readers share one page.
 *
 * Every writer fills its entire buffer with a single, thread-specific byte, so
 * a reader must always observe a complete page produced by one writer. If the
 * scheduler ever hands out a buffer that is concurrently modified, or the disk
 * manager copies partially, the read buffer would contain mixed bytes.
 */
// NOLINTNEXTLINE
TEST(DiskSchedulerThreadSafetyTest, DISABLED_ConcurrentSamePageNoTornReadTest) {
  constexpr int kNumThreads = 8;
  constexpr int kRounds = 16;
  constexpr page_id_t kPageId = 0;

  auto dm = std::make_unique<DiskManagerUnlimitedMemory>();
  auto disk_scheduler = std::make_unique<DiskScheduler>(dm.get());

  std::promise<void> start_promise;
  auto start_future = start_promise.get_future().share();
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int thread_id = 0; thread_id < kNumThreads; ++thread_id) {
    threads.emplace_back([&, thread_id] {
      start_future.wait();

      std::array<char, BUSTUB_PAGE_SIZE> write_buf;
      write_buf.fill(static_cast<char>(thread_id + 1));

      for (int round = 0; round < kRounds; ++round) {
        auto write_promise = disk_scheduler->CreatePromise();
        auto write_future = write_promise.get_future();
        std::vector<DiskRequest> write_requests;
        write_requests.push_back(DiskRequest{/*is_write=*/true, write_buf.data(), kPageId, std::move(write_promise)});
        disk_scheduler->Schedule(write_requests);
        if (!write_future.get()) {
          failures.fetch_add(1);
        }

        std::array<char, BUSTUB_PAGE_SIZE> read_buf{};
        auto read_promise = disk_scheduler->CreatePromise();
        auto read_future = read_promise.get_future();
        std::vector<DiskRequest> read_requests;
        read_requests.push_back(DiskRequest{/*is_write=*/false, read_buf.data(), kPageId, std::move(read_promise)});
        disk_scheduler->Schedule(read_requests);
        if (!read_future.get()) {
          failures.fetch_add(1);
        }

        const char observed = read_buf.front();
        const bool is_known_writer = observed >= 1 && observed <= kNumThreads;
        const bool is_uniform =
            std::all_of(read_buf.begin(), read_buf.end(), [observed](char byte) { return byte == observed; });
        if (!is_known_writer || !is_uniform) {
          failures.fetch_add(1);
        }
      }
    });
  }

  start_promise.set_value();
  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(failures.load(), 0);

  disk_scheduler.reset();
  dm->ShutDown();
}

/**
 * @brief Destroying the scheduler with outstanding requests must not lose them.
 *
 * The destructor is documented to signal the background thread to exit by
 * enqueueing `std::nullopt`. Since the request queue is FIFO, every request
 * that was scheduled before destruction should still complete.
 */
// NOLINTNEXTLINE
TEST(DiskSchedulerThreadSafetyTest, DISABLED_DestructorDrainsPendingRequestsTest) {
  constexpr int kRequests = 64;

  auto dm = std::make_unique<DiskManagerUnlimitedMemory>();
  auto disk_scheduler = std::make_unique<DiskScheduler>(dm.get());

  std::vector<std::array<char, BUSTUB_PAGE_SIZE>> buffers(kRequests);
  std::vector<std::future<bool>> futures;
  futures.reserve(kRequests);

  for (int i = 0; i < kRequests; ++i) {
    std::snprintf(buffers[i].data(), buffers[i].size(), "pending-page-%d", i);

    auto promise = disk_scheduler->CreatePromise();
    futures.push_back(promise.get_future());

    std::vector<DiskRequest> requests;
    requests.push_back(DiskRequest{/*is_write=*/true, buffers[i].data(), i, std::move(promise)});
    disk_scheduler->Schedule(requests);
  }

  // Destroy the scheduler before waiting on any future.
  disk_scheduler.reset();

  for (int i = 0; i < kRequests; ++i) {
    EXPECT_TRUE(futures[i].get());
  }

  dm->ShutDown();
}

}  // namespace bustub
