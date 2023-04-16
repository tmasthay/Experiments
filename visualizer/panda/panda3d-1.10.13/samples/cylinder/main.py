from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, LVector3, NodePath, GeomVertexFormat, GeomVertexData, GeomTriangles, Geom, GeomNode, Material, Vec3
from panda3d.bullet import BulletCylinderShape
from direct.interval.IntervalGlobal import LerpFunc
from panda3d.core import GeomVertexWriter
import numpy as np
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, GeomTristrips, PointLight

from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletBoxShape, BulletCylinderShape
from panda3d.bullet import BulletDebugNode, BulletSphereShape, BulletConvexHullShape
from panda3d.bullet import BulletCharacterControllerNode
from panda3d.bullet import BulletTriangleMesh, BulletTriangleMeshShape
from panda3d.bullet import BulletGhostNode, BulletBoxShape
from panda3d.core import CollisionTraverser, CollisionHandlerEvent, CollisionNode, CollisionSphere, CollisionBox, LPoint3f, WindowProperties



class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        self.setBackgroundColor(0, 0, 0)

        # Define materials for the cube
        mat_cube = Material()
        mat_cube.setAmbient((1, 1, 1, 1))
        mat_cube.setSpecular((0.2, 0.2, 0.2, 1))

        # Create the cube and add the material to it
        self.cube = self.create_cube(1)
        self.cube.set_pos(1, 0, 0)
        self.cube.set_hpr(0, 0, 0)
        self.cube.setTwoSided(True)
        self.cube.reparent_to(self.render)

        radius = 0.1
        height = 10
        shape1 = BulletCylinderShape(radius, height, 2)
        shape2 = BulletCylinderShape(Vec3(radius, 0, 0.5 * height), 2)

        # Create a cylinder
        self.cylinder_node = self.create_cylinder(radius, height, 16)
        # self.cylinder_node = shape1
        self.cylinder_node.reparentTo(self.render)
        self.cylinder_node.setTwoSided(True)

        texture = self.loader.loadTexture('laser_texture_red.png')
        self.cylinder_node.setTexture(texture)

        # Set the material properties
        mat = Material()
        mat.setSpecular((1, 1, 1, 1))
        mat.setAmbient((1,0,0,1))
        mat.setShininess(200)
        self.cylinder_node.setMaterial(mat)

        # Create a directional light pointing in the negative Z direction
        # self.cam.setPos(0, -20, -10)
        self.cam.setPos(0,-20,10)
        self.cam.lookAt(self.cylinder_node)
        base.camLens.setNearFar(0.1, 10000)

        # Set up the key bindings
        self.delta = 0.1
        self.accept('arrow_up', self.move_camera_forward)
        self.accept('arrow_down', self.move_camera_backward)


        self.dragging = False
        self.last_mouse_pos = None

        props = WindowProperties()
        props.set_cursor_hidden(False)
        self.win.request_properties(props)

        # self.accept("mouse1", self.start_dragging)
        # self.accept("mouse1-up", self.stop_dragging)
        # self.accept("aspectRatioChanged", self.adjust_aspect_ratio)
        # self.task_mgr.add(self.drag_camera_task, "drag_camera_task")


        self.position_text = OnscreenText(
            text='', 
            parent=base.a2dTopLeft,
            scale=0.05, fg=(1, 1, 1, 1),
            pos=(0.05, -0.08), 
            align=TextNode.ALeft
        )
        self.direction_text = OnscreenText(
            text='', 
            parent=base.a2dTopLeft,
            scale=0.05, 
            fg=(1, 1, 1, 1),
            pos=(0.05, -0.14), 
            align=TextNode.ALeft
        )

        self.taskMgr.add(self.update_text, "update_text")

        # Set up lighting
        ambient_light = AmbientLight("ambient_light")
        ambient_light.setColor((0.5, 0.5, 0.5, 1))  # White ambient light with strength 0.5
        directional_light = DirectionalLight("directional_light")
        directional_light.setDirection(LVector3(0, 0, -1))
        directional_light.setColor((0.9, 0.8, 0.9, 1))


        self.lights = []
        num_lights = 10
        for i in range(num_lights):
            light_node = PointLight("point_light_" + str(i))
            light_node.setColor((1, 1, 1, 1))
            light_position = Vec3(0, 0, i / (num_lights - 1) * height)
            attenuation = LVector3(0.1, 0.05, 0.02)
            light_node.setAttenuation(attenuation)
            light_node_path = self.cylinder_node.attachNewNode(light_node)
            light_node_path.setPos(light_position)
            self.render.setLight(light_node_path)
            self.lights.append(light_node_path)

        render.setLight(render.attachNewNode(ambient_light))
        render.setLight(render.attachNewNode(directional_light))

        # Animate the cylinder moving along the Y-axis
        def update_cylinder_position(t):
            self.cylinder_node.setPos(0.5, 0.5, 10 - t * 100)
            # self.cam.lookAt(self.cylinder_node)
            # alpha = 3 * t * 360
            # self.cylinder_node.setHpr(alpha, alpha, alpha)

        self.cylinder_move_interval = LerpFunc(
            update_cylinder_position,
            fromData=0,
            toData=1,
            duration=50.0,
            blendType='noBlend',
            extraArgs=[],
            name=None
        )

        self.theta = 0.0
        self.phi = 0.0
        self.drag_start = False
        self.cube_collision = self.create_cube_collision(1)
        self.cylinder_collision = self.create_cylinder_collision(radius, height)
        self.collision_traverser = CollisionTraverser()
        self.collision_handler = CollisionHandlerEvent()
        self.collision_handler.addInPattern('into-%in')
        self.collision_traverser.add_collider(self.cube_collision, self.collision_handler)
        self.accept('into-cube_collision', self.handle_intersection)
        self.taskMgr.add(self.check_for_intersection, "CheckForIntersectionTask")

        self.taskMgr.doMethodLater(0.5, self.strobe_effect, "StrobeEffectTask")
        self.cylinder_move_interval.loop()

    # def create_cylinder(self, radius, height, num_segments=16):
    #     format = GeomVertexFormat.getV3n3()
    #     vdata = GeomVertexData('cylinder', format, Geom.UHDynamic)

    #     vertex = GeomVertexWriter(vdata, 'vertex')
    #     normal = GeomVertexWriter(vdata, 'normal')

    #     for i in range(num_segments):
    #         angle = i / num_segments * 2 * 3.14159265
    #         x = radius * np.cos(angle)
    #         y = radius * np.sin(angle)
    #         for z in (0, height):
    #             print('(%f,%f)'%(x,y))

    #             vertex.addData3(x, y, z)
    #             normal.addData3(x, y, 0)

    #     geom = Geom(vdata)
    #     tris = GeomTriangles(Geom.UHDynamic)

    #     for i in range(num_segments):
    #         j = (i + 1) % num_segments
    #         tris.addVertices(i * 2, j * 2, j * 2 + 1)
    #         tris.addVertices(i * 2, j * 2 + 1, i * 2 + 1)

    #     geom.addPrimitive(tris)
    #     node = GeomNode('cylinder')
    #     node.addGeom(geom)

    #     return NodePath(node)
    def create_cylinder(self, radius, height, num_segments=16):
        format = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData('cylinder', format, Geom.UHDynamic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')
        texcoord = GeomVertexWriter(vdata, 'texcoord')

        bottom_center_vertex = vdata.get_num_rows()
        vertex.addData3(0, 0, 0)
        normal.addData3(0, 0, -1)
        texcoord.addData2(0.5, 0.5)

        top_center_vertex = vdata.get_num_rows()
        vertex.addData3(0, 0, height)
        normal.addData3(0, 0, 1)
        texcoord.addData2(0.5, 0.5)

        bottom_vertex = []
        top_vertex = []

        for i in range(num_segments):
            angle = i / num_segments * 2 * 3.14159265
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            u = i / num_segments

            for z in (0, height):
                v = z / height

                vertex.addData3(x, y, z)
                normal.addData3(x, y, 0)
                texcoord.addData2(u, v)

                print('(%f,%f,%f)'%(x,y,z))

                if z == 0:
                    bottom_vertex.append(vdata.get_num_rows() - 1)
                else:
                    top_vertex.append(vdata.get_num_rows() - 1)

        geom = Geom(vdata)
        prim = GeomTriangles(Geom.UHDynamic)

        # Add cylinder sides
        for i in range(num_segments):
            j = (i + 1) % num_segments
            prim.addVertices(bottom_vertex[i], top_vertex[j], top_vertex[i])
            prim.addVertices(bottom_vertex[i], bottom_vertex[j], top_vertex[j])

        # Add bottom cap
        for i in range(num_segments):
            j = (i + 1) % num_segments
            prim.addVertices(bottom_center_vertex, bottom_vertex[j], bottom_vertex[i])

        # Add top cap
        for i in range(num_segments):
            j = (i + 1) % num_segments
            prim.addVertices(top_center_vertex, top_vertex[i], top_vertex[j])

        geom.addPrimitive(prim)
        node = GeomNode('cylinder')
        node.addGeom(geom)

        return NodePath(node)


    def move_camera(self, x, y):
        self.cam.setPos(
            self.cam.getX() + x, 
            self.cam.getY() + y, 
            self.cam.getZ()
        )

    def update_text(self, task):
        pos = self.cam.getPos()
        hpr = self.cam.getHpr()
        self.position_text.setText(
            f'Camera position: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}'
        )
        self.direction_text.setText(
            f'Camera direction: h={hpr[0]:.2f}, p={hpr[1]:.2f}, r={hpr[2]:.2f}'
        )
        return task.cont
    
    def strobe_effect(self, task):
        hidden = self.cylinder_node.is_hidden()
        # if( hidden ):
        #     self.cylinder_node.show()
        # else:
        #     self.cylinder_node.show()
        return task.again
    
    def create_cube(self, size=1):
        format = GeomVertexFormat.getV3n3()
        vdata = GeomVertexData('cube', format, Geom.UHDynamic)

        vertex = GeomVertexWriter(vdata, 'vertex')
        normal = GeomVertexWriter(vdata, 'normal')

        # Define the vertices and normals for each face
        vertices = [
            # Front face
            (-size, -size,  size), ( size, -size,  size),
            ( size,  size,  size), (-size,  size,  size),

            # Back face
            ( size, -size, -size), (-size, -size, -size),
            (-size,  size, -size), ( size,  size, -size),

            # Right face
            ( size, -size,  size), ( size, -size, -size),
            ( size,  size, -size), ( size,  size,  size),

            # Left face
            (-size, -size, -size), (-size, -size,  size),
            (-size,  size,  size), (-size,  size, -size),

            # Top face
            (-size,  size,  size), ( size,  size,  size),
            ( size,  size, -size), (-size,  size, -size),

            # Bottom face
            (-size, -size, -size), ( size, -size, -size),
            ( size, -size,  size), (-size, -size,  size),
        ]

        normals = [
            ( 0,  0,  1),  # Front face
            ( 0,  0, -1),  # Back face
            ( 1,  0,  0),  # Right face
            (-1,  0,  0),  # Left face
            ( 0,  1,  0),  # Top face
            ( 0, -1,  0),  # Bottom face
        ]

        # Add the vertices and normals to the vertex buffer
        for i in range(6):
            for j in range(4):
                vertex.addData3f(*vertices[i*4 + j])
                normal.addData3f(*normals[i])

        # Define the triangles for each face
        geom = Geom(vdata)
        tris = GeomTriangles(Geom.UHDynamic)

        for i in range(6):
            tris.addVertices(i*4, i*4 + 1, i*4 + 2)
            tris.addVertices(i*4, i*4 + 2, i*4 + 3)

        geom.addPrimitive(tris)
        node = GeomNode('cube')
        node.addGeom(geom)

        return NodePath(node)
    
    def create_cube_collision(self, size=1):
        cube_collision = CollisionBox((0, 0, 0), size)
        collision_node = CollisionNode('cube_collision')
        collision_node.addSolid(cube_collision)
        collision_node.setIntoCollideMask(0)
        collision_node_path = self.cube.attachNewNode(collision_node)
        return collision_node_path
    
    def create_cylinder_collision(self, radius, height):
        cylinder_collision = CollisionSphere((0, 0, height / 2), radius)
        collision_node = CollisionNode('cylinder_collision')
        collision_node.addSolid(cylinder_collision)
        collision_node.setIntoCollideMask(0)
        collision_node_path = self.cylinder_node.attachNewNode(collision_node)
        return collision_node_path
    
    def handle_intersection(self, entry):
        mat = Material()
        mat.setAmbient((1, 1, 0, 1))
        mat.setDiffuse((1, 1, 0, 1))
        mat.setSpecular((0.2, 0.2, 0.2, 1))
        self.cube.setMaterial(mat)
        print('Intersection!')

    def check_for_intersection(self, task):
        self.collision_traverser.traverse(self.render)
        self.collision_traverser.addCollider(self.cylinder_collision, self.collision_handler)
        return task.cont
    
    def move_camera_forward(self):
        p = self.cam.getPos()
        a = 1 - self.delta
        print(p)
        self.cam.set_pos(LPoint3f(a*p[0], a*p[1], a*p[2]))

    def move_camera_backward(self):
        p = self.camera.get_pos()
        a = 1 + self.delta
        print(p)
        self.cam.set_pos(LPoint3f(a*p[0], a*p[1], a*p[2]))

    def start_dragging(self):
        self.dragging = True
        self.last_mouse_pos = self.mouseWatcherNode.get_mouse()

    def stop_dragging(self):
        self.dragging = False
        self.last_mouse_pos = None

    def drag_camera_task(self, task):
        if self.dragging and self.mouseWatcherNode.has_mouse():
            curr_mouse_pos = self.mouseWatcherNode.get_mouse()
            if self.last_mouse_pos is not None:
                delta = curr_mouse_pos - self.last_mouse_pos
                self.rotate_camera(delta)
            self.last_mouse_pos = curr_mouse_pos
        return task.cont

    def rotate_camera(self, delta):
        h_delta, p_delta = delta.x * 100, delta.y * 100
        # self.camera.set_hpr(self.camera.get_h() - h_delta, self.camera.get_p() + p_delta, 0)
        diff = self.cube.get_pos() - self.camera.get_pos()

        print(self.cube.get_pos())

        r = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
        self.theta += self.delta
        self.phi += self.delta
        p = self.cube.getPos()
        x_new = p[0] + r * p[0] * np.cos(self.phi) * np.sin(self.theta)
        y_new = p[1] + r * p[1] * np.sin(self.phi) * np.sin(self.theta)
        z_new = p[2] + r * p[2] * np.cos(self.theta)
        self.cam.setPos(LPoint3f(x_new, y_new, z_new))
        self.camera.look_at(p)

    def adjust_aspect_ratio(self):
        aspect_ratio = self.get_aspect_ratio()
        self.camera.node().get_lens().set_aspect_ratio(aspect_ratio)


app = MyApp()
app.run()

