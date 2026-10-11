// Host-tier metrics formatting: the /metrics wire contract. Builds a known snapshot and checks
// both the Prometheus text exposition and the JSON object carry the device, KV, host, and
// scheduler figures. No GPU or artifact required.

#include "serve/metrics_format.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::MetricsSnapshot sample() {
    ninfer::MetricsSnapshot m;
    m.scheduler.running_requests      = 2;
    m.scheduler.prefilling_requests   = 1;
    m.scheduler.decode_ready_requests = 2;
    m.scheduler.waiting_requests      = 3;
    m.scheduler.computed_prefill_tokens = 1000;
    m.scheduler.committed_decode_tokens = 2000;
    m.scheduler.decode_rounds           = 50;
    m.scheduler.decode_row_rounds       = 75;
    m.devices.push_back(ninfer::DeviceMemoryMetrics{
        .device = 4, .total_bytes = 16ULL << 30, .free_bytes = 6ULL << 30, .used_bytes = 10ULL << 30});
    m.main_kv = ninfer::KvPoolMetrics{.capacity_pages = 16384,
                                      .mapped_pages = 100,
                                      .cached_pages = 20,
                                      .free_pages = 16264,
                                      .page_bytes = 2ULL << 20};
    m.prefix_cache.enabled                             = true;
    m.prefix_cache.device_snapshot_slots               = 5;
    m.prefix_cache.device_snapshot_slots_used          = 2;
    m.prefix_cache.host.enabled                        = true;
    m.prefix_cache.host.pinned_bytes                   = 28130ULL << 20;
    m.prefix_cache.host.used_bytes                     = 7ULL << 30;
    m.prefix_cache.host.block_slab_count               = 10291;
    m.prefix_cache.host.snapshot_slab_count            = 38;
    return m;
}

int test_prometheus() {
    const std::string text = ninfer::serve::metrics_to_prometheus(sample());
    int failures           = 0;
    const auto has         = [&](const std::string& needle) {
        return text.find(needle) != std::string::npos;
    };
    failures += check(has("# TYPE ninfer_device_memory_bytes gauge"),
                      "device metric type missing");
    failures += check(has("ninfer_device_memory_bytes{device=\"4\",kind=\"used\"} " +
                          std::to_string(10ULL << 30)),
                      "device used bytes sample missing");
    failures += check(has("ninfer_kv_pages{pool=\"main\",kind=\"cached\"} 20"),
                      "main KV cached pages missing");
    failures += check(has("ninfer_kv_page_bytes{pool=\"main\"} 2097152"),
                      "KV page bytes missing");
    failures += check(has("ninfer_prefix_cache_device_snapshot_slots{kind=\"used\"} 2"),
                      "device snapshot used missing");
    failures += check(has("ninfer_prefix_cache_host_bytes{kind=\"pinned\"} " +
                          std::to_string(28130ULL << 20)),
                      "host pinned bytes missing");
    failures += check(has("ninfer_prefix_cache_host_slabs{kind=\"block\"} 10291"),
                      "host block slab count missing");
    failures += check(has("ninfer_requests{state=\"waiting\"} 3"), "waiting requests missing");
    failures += check(has("# TYPE ninfer_decode_rounds_total counter"),
                      "decode rounds counter missing");
    return failures;
}

int test_json() {
    int failures = 0;
    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(ninfer::serve::metrics_to_json(sample()).dump());
    } catch (const std::exception& e) {
        std::cerr << "json parse failed: " << e.what() << '\n';
        return 1;
    }
    failures += check(parsed["devices"][0]["device"] == 4, "json device ordinal wrong");
    failures += check(parsed["devices"][0]["used_bytes"] == (10ULL << 30),
                      "json device used wrong");
    failures += check(parsed["main_kv"]["capacity_pages"] == 16384,
                      "json kv capacity wrong");
    failures += check(parsed["prefix_cache"]["host"]["pinned_bytes"] == (28130ULL << 20),
                      "json host pinned wrong");
    failures += check(parsed["prefix_cache"]["host"]["block_slab_count"] == 10291,
                      "json host slab count wrong");
    failures += check(parsed["scheduler"]["waiting_requests"] == 3,
                      "json scheduler waiting wrong");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_prometheus();
    failures += test_json();
    if (failures == 0) {
        std::printf("metrics format tests passed.\n");
        return 0;
    }
    std::printf("%d metrics format check(s) failed.\n", failures);
    return 1;
}
