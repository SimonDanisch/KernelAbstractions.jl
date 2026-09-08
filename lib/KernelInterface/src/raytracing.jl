# Ray-tracing intrinsics a shader body calls.
#
# Beside `device.jl`, and for the same reason: these are device-side names that
# a COMPILER lowers, so they belong under everything that compiles or runs a
# shader rather than in one of them.
#
# They were `lava_rt_*` in Lava, defined there as Lava's own functions. Two
# consequences, both of which this file removes:
#
#   * The vendor was in the NAME, which a code path may not carry.
#   * A renderer that wanted them had to name a backend. Hikari reaches the GPU
#     through Mantle for everything else and still had `import Lava` at the top
#     of `rt-pipeline.jl` solely for these — and when Lava took a dependency on
#     Vulkan again, that one line stopped Hikari loading on any machine without
#     a Vulkan driver.
#
# Lava cannot override `Mantle.<name>`, because Mantle depends on Lava and not
# the other way round; that is why a bridge existed at all. Both depend on
# KernelInterface, so here each backend overrides directly and there is no
# bridge to keep in step.
#
# The set is what CALLERS use, taken from Hikari's call sites rather than from
# what one backend happened to implement: `rt_trace_ray`,
# `rt_payload_store_f32_at`, `rt_payload_load_f32_at` and `rt_terminate_ray`
# were imported there and never called, so they are not vocabulary.
#
# A backend that cannot answer one says so through `caps`, not by leaving a
# `MethodError` for a shader to find at compile time.

const _RT_OUTSIDE = " used outside a ray-tracing shader, or not captured by a " *
                    "compiler that lowers it"

"""
    rt_launch_id_x()::UInt32

This invocation's index along x in the ray-tracing launch grid.

`gl_LaunchIDEXT.x` in GLSL. The one-dimensional launch is the shape Mantle's
trace passes use: a ray index, not a pixel.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override rt_launch_id_x()::UInt32
    ```
"""
function rt_launch_id_x end

"""
    rt_hit_object_trace_ray(flags, cullmask, sbt_offset, sbt_stride, miss_index,
                            ox, oy, oz, tmin, dx, dy, dz, tmax)

Trace a ray and record the hit WITHOUT invoking its shader.

The hit is held in the invocation's hit object; `rt_reorder_thread` may then
regroup invocations by what they hit, and `rt_hit_object_execute_shader` runs
the shader for it. Splitting the three is the point — it is what lets divergent
hits be sorted before any shading happens.

Thirteen arguments and no struct, because a shader passes them as immediates
and every one is a separate operand of the instruction this lowers to.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override rt_hit_object_trace_ray(::UInt32, ::UInt32, ::UInt32, ::UInt32,
                                             ::UInt32, ::Float32, ::Float32, ::Float32,
                                             ::Float32, ::Float32, ::Float32, ::Float32,
                                             ::Float32)
    ```
"""
function rt_hit_object_trace_ray end

"""
    rt_reorder_thread()

Regroup invocations by the hit each is holding, so the shader that follows runs
over coherent work.

A hint: an implementation that reorders nothing is correct and slower.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override rt_reorder_thread()
    ```
"""
function rt_reorder_thread end

"""
    rt_hit_object_execute_shader()

Run the shader for the hit this invocation is holding.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override rt_hit_object_execute_shader()
    ```
"""
function rt_hit_object_execute_shader end

"""
    rt_ignore_intersection()

Discard the intersection the any-hit shader is being run for, and carry on
traversing. Only callable from an any-hit shader.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override rt_ignore_intersection()
    ```
"""
function rt_ignore_intersection end

# ── What the hit was ─────────────────────────────────────────────────────────
#
# One accessor per value rather than a struct: they lower to separate
# instructions, and a shader that reads only the primitive id should not pay
# for the barycentrics.

"""
    rt_primitive_id()::UInt32

Index of the primitive that was hit, within its geometry. One-based, like every
index Mantle hands a shader.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override rt_primitive_id()::UInt32
    ```
"""
function rt_primitive_id end

"""
    rt_instance_id()::UInt32

Index of the instance that was hit, within the top-level structure.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override rt_instance_id()::UInt32
    ```
"""
function rt_instance_id end

"""
    rt_instance_custom_index()::UInt32

The value the instance was built with, rather than its position in the
structure — what a renderer puts its own mesh or material id in.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override rt_instance_custom_index()::UInt32
    ```
"""
function rt_instance_custom_index end

"""
    rt_ray_tmax()::Float32

Distance along the ray at which it hit, or the ray's `t_max` when it missed.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override rt_ray_tmax()::Float32
    ```
"""
function rt_ray_tmax end

"""
    rt_hit_bary_u()::Float32
    rt_hit_bary_v()::Float32

The two barycentric coordinates of the hit within its triangle. The third is
`1 - u - v` and is not an intrinsic, because no API supplies it.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override rt_hit_bary_u()::Float32
    @device_override rt_hit_bary_v()::Float32
    ```
"""
function rt_hit_bary_u end
@doc (@doc rt_hit_bary_u) function rt_hit_bary_v end

# The host answers. They exist to SAY something: a bare `MethodError` on a
# zero-argument function reads as "this method is missing", not as "you have
# called a GPU instruction on the CPU".
for f in (:rt_launch_id_x, :rt_reorder_thread, :rt_hit_object_execute_shader,
          :rt_ignore_intersection, :rt_primitive_id, :rt_instance_id,
          :rt_instance_custom_index, :rt_ray_tmax, :rt_hit_bary_u, :rt_hit_bary_v)
    @eval $f(args...) = error($(string(f)) * _RT_OUTSIDE)
end
rt_hit_object_trace_ray(args...) = error("rt_hit_object_trace_ray" * _RT_OUTSIDE)
