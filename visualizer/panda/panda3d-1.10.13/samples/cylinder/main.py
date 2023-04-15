from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight, LVector3, NodePath, GeomVertexFormat, GeomVertexData, GeomTriangles, Geom, GeomNode, Material, Vec3
from panda3d.bullet import BulletCylinderShape
from direct.interval.IntervalGlobal import LerpFunc
from panda3d.core import GeomVertexWriter
import numpy as np
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, GeomTristrips



class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        self.setBackgroundColor(0, 0, 0)

        radius = 0.1
        height = 100
        shape1 = BulletCylinderShape(radius, height, 2)
        shape2 = BulletCylinderShape(Vec3(radius, 0, 0.5 * height), 2)

        # Create a cylinder
        cylinder_node = self.create_cylinder(radius, height, 16)
        # cylinder_node = shape1
        cylinder_node.reparentTo(self.render)
        cylinder_node.setTwoSided(True)

        texture = self.loader.loadTexture('laser_texture_red.png')
        cylinder_node.setTexture(texture)

        # Set the material properties
        mat = Material()
        mat.setSpecular((1, 1, 1, 1))
        mat.setAmbient((1,0,0,1))
        mat.setShininess(200)
        cylinder_node.setMaterial(mat)

        # Create a directional light pointing in the negative Z direction
        self.cam.setPos(0, -20, -10)
        self.cam.lookAt(cylinder_node)
        base.camLens.setNearFar(0.1, 10000)

        # Set up the key bindings
        self.accept('arrow_up', self.move_camera, [0, 1])
        self.accept('arrow_down', self.move_camera, [0, -1])
        self.accept('arrow_left', self.move_camera, [-1, 0])
        self.accept('arrow_right', self.move_camera, [1, 0])

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

        render.setLight(render.attachNewNode(ambient_light))
        render.setLight(render.attachNewNode(directional_light))

        # Animate the cylinder moving along the Y-axis
        def update_cylinder_position(t):
            cylinder_node.setPos(0, t * 100, height / 2)
            self.cam.lookAt(cylinder_node)
            alpha = 3 * t * 360
            cylinder_node.setHpr(alpha, alpha, alpha)

        self.cylinder_move_interval = LerpFunc(
            update_cylinder_position,
            fromData=0,
            toData=1,
            duration=50.0,
            blendType='noBlend',
            extraArgs=[],
            name=None
        )
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


app = MyApp()
app.run()

