"""
Primitive topology: how a vertex stream is grouped into primitives.

Here for the same reason [`MatrixShape`](@ref) and [`DeviceCaps`](@ref) are.
Two sides need this vocabulary and neither can own it:

  * a **compiler** dispatches on it to emit the geometry stage's input
    execution mode — `TriangleList` → three vertices per invocation,
    `LineListAdjacency` → four;
  * a **backend** dispatches on it to create a pipeline —
    `VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST`, `MTLPrimitiveTypeTriangle`.

It lived in Lava, which is a Julia→SPIR-V compiler, and a Metal backend needs
`TriangleList` without importing one. Defining it twice was the alternative, and
that is exactly what `DeviceCaps` did before it moved here — two declarations
bridged by a positional copy, which drifted.

Nothing here is any one API's. The names are the ones every graphics API uses.
"""
abstract type Topology end

"Three vertices per triangle, disjoint groups."
struct TriangleList <: Topology end

"A sliding three-wide window: `n` vertices give `n-2` triangles."
struct TriangleStrip <: Topology end

"Two vertices per line, disjoint groups."
struct LineList <: Topology end

"A sliding two-wide window: `n` vertices give `n-1` segments."
struct LineStrip <: Topology end

"One vertex per point."
struct PointList <: Topology end

"Patches for the tessellation stages."
struct PatchList <: Topology end

# Both feed a geometry shader four vertices per primitive. They differ in how the
# index buffer is walked: the LIST form consumes a disjoint group of 4 per
# primitive, the STRIP form slides a 4-wide window one index at a time. An
# adjacency index list built for a strip (Makie's polylines: `0 0 1 2 3 3`)
# yields ⌊n/4⌋ primitives under the list form instead of n-3 — a polyline drawn
# as scattered dashes.
struct LineListAdjacency <: Topology end
struct LineStripAdjacency <: Topology end
