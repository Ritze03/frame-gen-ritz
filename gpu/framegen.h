// framegen: image-only frame interpolation (Vulkan compute port of proto/interp.py).
// Plain Vulkan, no gamescope types. Never submits, never touches queues.
//
// Threading / queues: not thread-safe. All records of one Interpolator must be submitted to ONE queue,
// in recording order: its internal buffers are shared between records and only ordered by pipeline
// barriers, which do not order against work on another queue.
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

namespace framegen {

struct InitInfo {
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;   // all device functions are loaded through this
    uint32_t width = 0, height = 0;           // >= kMinSize each
    // B8G8R8A8_UNORM or R8G8B8A8_UNORM only (8-bit, UNORM, never _SRGB). Anything else -> init false.
    VkFormat viewFormat = VK_FORMAT_B8G8R8A8_UNORM;
    // Required (instance-level, cannot come from gdpa): used to put the buffers in DEVICE_LOCAL memory.
    // init() returns false if null.
    PFN_vkGetPhysicalDeviceMemoryProperties getMemProps = nullptr;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
};

class Interpolator {
public:
    // false -> the object is not usable (record() is a no-op returning false); skip frame generation
    // for this surface. Fails for sizes below kMinSize, unsupported formats, null getMemProps, or
    // any Vulkan creation/allocation failure.
    // init() on a live object and destroy()/resize() free resources that earlier records use: the
    // caller must guarantee every previous record has retired first (fence/timeline wait, or
    // vkQueueWaitIdle/vkDeviceWaitIdle; resolution changes are rare).
    bool init(const InitInfo&);
    void destroy();
    // Re-creates the size-dependent buffers. Same retire rule as above. On false the object stays
    // alive but holds no size (record() returns false) until a later resize() succeeds.
    bool resize(uint32_t w, uint32_t h, VkFormat viewFormat);

    // Call once per command buffer / submission, before its first record*() call: resets the
    // per-submission counts of both descriptor rings (see "Descriptor rings" below).
    void beginFrame();

    // Frame generation for one real pair (prev, curr) is split in two:
    //   recordEstimate(prev, curr)        passes 1-8 (motion estimate), once per real pair
    //   recordSynth(prev, curr, out, t)   the generated frame at time t, any number of times after one
    //                                     recordEstimate, each into its own out (N x: t = k/N, k = 1..N-1)
    //   record(prev, curr, out, t)        = recordEstimate + recordSynth (2x: t = 0.5)
    //
    // All views: 2D, single mip/layer, the same size as init/resize, format viewFormat, i.e. 8-bit UNORM
    // (an _SRGB view gives wrong luma; 10-bit is unsupported: skip FG or convert first),
    // VK_IMAGE_LAYOUT_GENERAL on entry and exit.
    // prev/curr: SAMPLED usage, already made readable by the caller. out: STORAGE usage, caller-owned.
    //
    // recordEstimate ends with a COMPUTE -> COMPUTE barrier on the library's own buffers. recordSynth ends
    // with a COMPUTE_SHADER SHADER_WRITE -> COMPUTE_SHADER SHADER_READ memory barrier (valid on a
    // compute-only queue). Other consumers of out (fragment, transfer) need the caller's own barrier.
    // Synths of one pair share internal scratch (field at t, mismatch, counter) and are serialised by
    // barriers, in recording order; each one's out is complete when its own record ends.
    //
    // t in (0, 1). t == 0.5 is exactly the 2x midpoint path. Every fallback (per pixel, scene cut, more than
    // 15 % untrusted) presents the nearer real frame: prev for t <= 0.5, curr for t > 0.5.
    //
    // Descriptor rings: recordEstimate takes the next of kSetRing estimate sets, recordSynth the next of
    // kSetRing synth sets; record() takes one of each. A call returns false and records nothing (out
    // untouched, so the caller must not substitute it) if the object is not initialised, t is outside (0, 1),
    // or its ring already gave out kSetRing sets since the last beginFrame() (asserts in debug builds).
    // The caller must also keep the estimates and the synths of all not-yet-retired submissions
    // <= kSetRing each, in total.
    // recordSynth also returns false if no recordEstimate was recorded since init/resize, or if prev/curr
    // are not the views of the latest recordEstimate (it always uses the latest estimate).
    bool recordEstimate(VkCommandBuffer cb, VkImageView prev, VkImageView curr);
    bool recordSynth(VkCommandBuffer cb, VkImageView prev, VkImageView curr, VkImageView out, float t);
    bool record(VkCommandBuffer cb, VkImageView prev, VkImageView curr, VkImageView out, float t = 0.5f);

    // Profiling (test/benchmark only): timestamps after every pass group. recordEstimate (or record)
    // starts a new profile, recordSynth appends its passes to it. Returns false if the query pool cannot
    // be created (profiling then stays off).
    bool setProfiling(bool on);
    // Reads the last profile (waits for results). Fills names + GPU ticks per pass group
    // (multiply by VkPhysicalDeviceLimits::timestampPeriod for ns). Returns false if unavailable.
    bool readTimings(std::vector<const char*>& names, std::vector<uint64_t>& ticks);

    static constexpr uint32_t kSetRing = 32;
    static constexpr uint32_t kMinSize = 16;
    struct Impl;
private:
    Impl* m = nullptr;
};

}  // namespace framegen
