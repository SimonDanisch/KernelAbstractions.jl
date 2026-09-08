# The graphics stage intrinsics a shader body calls.
#
# Beside `device.jl` and `raytracing.jl`, for the reason those two are here:
# a device-side name that a COMPILER lowers belongs under everything that
# compiles or runs a shader.
#
# These were declared in Mantle and defined AGAIN in Lava, with a bridge in
# `MantleVulkanExt` — `Mantle.$f() = Lava.$f()` — that existed only because Lava
# cannot override a Mantle name: Mantle depends on Lava, not the other way
# round. Both depend on KernelInterface. So each backend overrides here, the
# bridge is gone, and a downstream package imports one name from one place.
#
# What is NOT here, and deliberately:
#
#   * `set_position!`, `gfx_output`/`gfx_input` and their flat variants,
#     `geom_input`, `geom_input_position` — the older location-based way of
#     moving values between stages. A pipeline declares `varyings = (albedo =
#     Vec4f, …)`, a vertex stage returns `(position = …, albedo = …)` and a
#     fragment stage receives them by name. That is one declaration instead of
#     matched pairs of numbered slots, it works on both backends today, and
#     `bench/showcase.jl` is written in it. Seven names removed rather than
#     moved.
#   * A geometry stage's `emit_vertex!`/`end_primitive!`. Metal has no geometry
#     shaders at all — Apple's replacement is the mesh pipeline, which Mantle
#     does not describe — so they are declared but a backend answers whether it
#     has them through `caps`, and a pipeline naming one is refused when it
#     does not.
#
# Indices are ONE-BASED on every backend. A shader that has to remember which
# convention a builtin follows will eventually get it wrong.

const _GFX_OUTSIDE = " is a shader builtin: it reads a value the rasteriser " *
                     "supplies, and only a compiled stage has one. Calling it " *
                     "on the host cannot mean anything."

"""
    vertex_index()::Int32

Which vertex this invocation is for, counting from one.

`gl_VertexIndex` in GLSL, `[[vertex_id]]` in MSL. Vertex stage only.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override vertex_index()::Int32
    ```
"""
function vertex_index end

"""
    instance_index()::Int32

Which instance this invocation is for, counting from one.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override instance_index()::Int32
    ```
"""
function instance_index end

"""
    frag_coord(dim = 1)::Float32

One component of the interpolated fragment position, counting from one: 1 is x,
4 is w. `gl_FragCoord` in GLSL, `[[position]]` in MSL. Fragment stage only.

`frag_coord_x`, `frag_coord_y`, `frag_coord_z`, `frag_coord_w` and
`frag_coord_xy` name the components a shader usually wants, so the common case
needs no literal.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override frag_coord(dim::Integer = 1)::Float32
    ```
"""
function frag_coord end

"""x of [`frag_coord`](@ref)."""
function frag_coord_x end
"""y of [`frag_coord`](@ref)."""
function frag_coord_y end
"""z of [`frag_coord`](@ref), the depth this fragment writes."""
function frag_coord_z end
"""w of [`frag_coord`](@ref)."""
function frag_coord_w end
"""x and y of [`frag_coord`](@ref), as a tuple."""
function frag_coord_xy end

"""
    dFdx(v::Float32)::Float32
    dFdy(v::Float32)::Float32

How `v` changes across the fragment quad in x and in y. Fragment stage only:
the value is a difference between neighbouring invocations, so it exists only
where invocations run in a quad.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override dFdx(::Float32)::Float32
    @device_override dFdy(::Float32)::Float32
    ```
"""
function dFdx end
@doc (@doc dFdx) function dFdy end

"""
    set_point_size!(s::Float32)

The size, in pixels, of the point this vertex becomes. Only meaningful when the
pipeline's topology is a point list.

A setter and not a returned field, unlike `position` and the varyings, because
it is written by a minority of shaders and threading it through every vertex
stage's return type to carry `nothing` almost always is the worse trade.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override set_point_size!(::Float32)
    ```
"""
function set_point_size! end

"""
    sample_texture_2d(binding::UInt32, u::Float32, v::Float32, component::UInt32)::Float32

One component of a filtered sample from the 2D texture at `binding`, at
normalised coordinates `(u, v)`.

A component at a time, and a `UInt32` binding rather than a texture object,
because that is the shape both backends' bound-texture access has: the shader
names a slot, not a handle.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override sample_texture_2d(::UInt32, ::Float32, ::Float32, ::UInt32)::Float32
    ```
"""
function sample_texture_2d end

# ── Geometry stage ───────────────────────────────────────────────────────────
#
# Declared, and not every backend has them: Metal has no geometry stage. A
# backend says so through `caps` and a pipeline naming one is refused there,
# which is a question a caller can ask — unlike a `MethodError` from inside a
# shader compile, which is what the arrangement before this file gave.

"""
    emit_vertex!()

Emit the vertex the geometry stage has been writing.

!!! note
    Backend implementations with a geometry stage **must** implement:
    ```
    @device_override emit_vertex!()
    ```
"""
function emit_vertex! end

"""
    end_primitive!()

Finish the primitive the geometry stage has been emitting vertices for.

!!! note
    Backend implementations with a geometry stage **must** implement:
    ```
    @device_override end_primitive!()
    ```
"""
function end_primitive! end

"""
    primitive_id_in()::UInt32

Index of the primitive the geometry stage is being run for.

!!! note
    Backend implementations with a geometry stage **must** implement:
    ```
    @device_override primitive_id_in()::UInt32
    ```
"""
function primitive_id_in end

# ── Clip space ───────────────────────────────────────────────────────────────

"""
    clip_y(y::Float32)::Float32

`y` in this backend's clip space, given a `y` in the portable one.

**The portable clip space has +y pointing DOWN the screen**, which is Vulkan's.
A backend whose own points up overrides this with a negation, and the vertex
stage applies it to every `position` it writes.

It is NOT one of the builtins above and its default is the identity rather than
an error, for three reasons: it takes an argument, it is meaningful on the host,
and a backend that agrees with the portable convention should not have to say
so.

A shader that does the viewport transform BY HAND — a shadow lookup turning
`sunvp * world` into a texel row — has to apply the same mirror, which is what
this is exported for. It is its own inverse.

Getting the vertex stage's half right and this half wrong is not a visible
mistake: a mirrored scene still looks like a scene, its shadow map is mirrored
with it, and two renders from the same backend agree perfectly. What gave it
away in September 2026 was a person looking at a window.
"""
clip_y(y::Float32) = y

# The host answers, which exist to SAY something rather than to work.
for f in (:vertex_index, :instance_index, :frag_coord, :frag_coord_x, :frag_coord_y,
          :frag_coord_z, :frag_coord_w, :frag_coord_xy, :dFdx, :dFdy,
          :set_point_size!, :sample_texture_2d, :emit_vertex!, :end_primitive!,
          :primitive_id_in)
    @eval $f(args...) = error($(string(f)) * _GFX_OUTSIDE)
end
