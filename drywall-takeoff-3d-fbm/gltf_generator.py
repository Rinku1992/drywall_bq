import numpy as np
from pygltflib import (
    Material, PbrMetallicRoughness,
    GLTF2, Scene, Node, Mesh, Primitive,
    Buffer, BufferView, Accessor,
    FLOAT, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER
)

def add_materials(gltf):
    gltf.materials = list()

    wall_material = Material(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.9
        )
    )
    gltf.materials.append(wall_material)

    roof_material = Material(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[0.7, 0.7, 1.0, 0.2],
            metallicFactor=0.0,
            roughnessFactor=0.9
        ),
        alphaMode="BLEND",
        doubleSided=True
    )
    gltf.materials.append(roof_material)
    return dict(wall=0, roof=1)

def create_wall_vertices(x1, y1, x2, y2, height, thickness):
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    nx, ny = -dy / length, dx / length

    t = thickness / 2

    p1 = [x1 + nx*t, y1 + ny*t, 0]
    p2 = [x1 - nx*t, y1 - ny*t, 0]
    p3 = [x2 - nx*t, y2 - ny*t, 0]
    p4 = [x2 + nx*t, y2 + ny*t, 0]

    p5 = [*p1[:2], height]
    p6 = [*p2[:2], height]
    p7 = [*p3[:2], height]
    p8 = [*p4[:2], height]

    vertices = np.array([
        p1, p2, p3, p4,
        p5, p6, p7, p8
    ], dtype=np.float32)

    indices = np.array([
        0,1,5, 0,5,4,
        1,2,6, 1,6,5,
        2,3,7, 2,7,6,
        3,0,4, 3,4,7,
        4,5,6, 4,6,7,
        0,3,2, 0,2,1
    ], dtype=np.uint16)

    return vertices, indices

def create_roof_vertices(polygon_xy, height):
    vertices = np.array(
        [[x, y, height] for x, y in polygon_xy],
        dtype=np.float32
    )

    indices = list()
    for i in range(1, len(vertices) - 1):
        indices.extend([0, i, i + 1])

    return vertices, np.array(indices, dtype=np.uint16)

def add_mesh(gltf, buffer_data, buffer_views, accessors, vertices, indices, material):
    v_offset = len(buffer_data)
    buffer_data.extend(vertices.tobytes())

    i_offset = len(buffer_data)
    buffer_data.extend(indices.tobytes())

    v_view = BufferView(
        buffer=0,
        byteOffset=v_offset,
        byteLength=vertices.nbytes,
        target=ARRAY_BUFFER
    )

    i_view = BufferView(
        buffer=0,
        byteOffset=i_offset,
        byteLength=indices.nbytes,
        target=ELEMENT_ARRAY_BUFFER
    )

    v_accessor = Accessor(
        bufferView=len(buffer_views),
        componentType=FLOAT,
        count=len(vertices),
        type="VEC3",
        max=vertices.max(axis=0).tolist(),
        min=vertices.min(axis=0).tolist()
    )

    i_accessor = Accessor(
        bufferView=len(buffer_views) + 1,
        componentType=5123,
        count=len(indices),
        type="SCALAR"
    )

    buffer_views.extend([v_view, i_view])
    accessors.extend([v_accessor, i_accessor])

    mesh = Mesh(
        primitives=[Primitive(
            attributes={"POSITION": len(accessors) - 2},
            indices=len(accessors) - 1,
            material=material
        )]
    )

    gltf.meshes.append(mesh)
    gltf.nodes.append(Node(mesh=len(gltf.meshes) - 1))

def load_gltf(walls, polygons, output="/tmp/walls.gltf"):
    gltf = GLTF2()
    buffer_data = bytearray()
    buffer_views = list()
    accessors = list()
    gltf.meshes = list()
    gltf.nodes = list()
    materials = add_materials(gltf)

    for wall in walls:
        vertices, indices = create_wall_vertices(**wall)
        add_mesh(gltf, buffer_data, buffer_views, accessors, vertices, indices, materials["wall"])

    if polygons:
        for poly in polygons:
            verts_xy = poly["vertices"]
            height = poly["height"]

            vertices, indices = create_roof_vertices(verts_xy, height)

            if len(indices) == 0:
                continue

            add_mesh(gltf, buffer_data, buffer_views, accessors, vertices, indices, materials["roof"])

    gltf.buffers.append(Buffer(byteLength=len(buffer_data)))
    gltf.bufferViews = buffer_views
    gltf.accessors = accessors
    gltf.scenes = [Scene(nodes=list(range(len(gltf.nodes))))]
    gltf.scene = 0
    gltf.set_binary_blob(buffer_data)
    gltf.save(output)