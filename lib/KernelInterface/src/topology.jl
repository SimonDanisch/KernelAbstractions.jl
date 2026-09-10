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

"""
    primitivevertices(t::Topology) -> Int

How many vertices one primitive of `t` has.

The number a geometry stage is handed per invocation, and the number a mesh
lowering writes per primitive. It is a property of the TOPOLOGY and nothing
else, which is why it is here: it was written out twice — once in Lava's SPIR-V
emitter to pick the geometry stage's input execution mode, once in Mantle's
Vulkan backend to size the arrayed inputs its geometry wrapper reads — and two
tables of one fact drift the moment a topology is added to one of them.

A strip's answer is the size of its sliding WINDOW, not the length of the strip:
a `TriangleStrip` delivers three vertices per primitive like a `TriangleList`,
and the two differ in how the index stream is walked, not in what a primitive
is.
"""
function primitivevertices end

primitivevertices(::PointList)          = 1
primitivevertices(::LineList)           = 2
primitivevertices(::LineStrip)          = 2
primitivevertices(::TriangleList)       = 3
primitivevertices(::TriangleStrip)      = 3
primitivevertices(::LineListAdjacency)  = 4
primitivevertices(::LineStripAdjacency) = 4

# No answer, rather than a wrong one: a patch's vertex count is the tessellation
# configuration's (`TessConfig.patch_size`), not the topology's, and a caller
# that reaches here has asked the wrong thing.
primitivevertices(::PatchList) = throw(ArgumentError(
    "a patch's vertex count is the tessellation configuration's, not the topology's"))
