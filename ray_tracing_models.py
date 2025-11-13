# pyright: reportInvalidTypeForm=false
from operator import is_
import taichi as ti

PI = 3.14159265

@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])

@ti.func
def random_in_unit_sphere():
    p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    while p.norm() >= 1.0:
        p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    return p

@ti.func
def random_unit_vector():
    return random_in_unit_sphere().normalized()

@ti.func
def to_light_source(hit_point, light_source):
    return light_source - hit_point

@ti.func
def reflect(v, normal):
    return v - 2 * v.dot(normal) * normal

@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = ti.min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel

@ti.func
def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)

@ti.dataclass
class Ray:
    """Ray in 3D space."""
    origin: ti.types.vector(3, ti.f32)
    direction: ti.types.vector(3, ti.f32)
    
    @ti.func
    def at(self, t):
        """Compute the point along the ray at parameter t."""
        return self.origin + t * self.direction

@ti.data_oriented
class Sphere:
    """A sphere object in the scene.
    
    Parameters
    ----------
    center : ti.Vector
        The center of the sphere.
    radius : float
        The radius of the sphere.
    material : int
        The material type of the sphere.
    color : ti.Vector
        The color of the sphere.
    """
    def __init__(self, center, radius, material, color):
        self.center = center
        self.radius = radius
        self.material = material
        self.color = color

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        is_hit = False
        front_face = False
        root = 0.0
        hit_point =  ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        if discriminant > 0:
            sqrtd = ti.sqrt(discriminant)
            root = (-b - sqrtd) / (2 * a)
            if t_min <= root <= t_max:
                is_hit = True
            else:
                root = (-b + sqrtd) / (2 * a)
                if root >= t_min and root <= t_max:
                    is_hit = True
        if is_hit:
            hit_point = ray.at(root)
            hit_point_normal = (hit_point - self.center) / self.radius
            # Check which side does the ray hit, we set the hit point normals always point outward from the surface
            if ray.direction.dot(hit_point_normal) < 0:
                front_face = True
            else:
                hit_point_normal = -hit_point_normal
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

@ti.data_oriented
class Hittable_list:
    """A list of hittable objects in the scene."""
    def __init__(self):
        self.objects = []
    def add(self, obj):
        """Add a hittable object to the list."""
        self.objects.append(obj)
    def clear(self):
        """Remove all hittable objects from the list."""
        self.objects = []

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        """Check for the closest hit of the ray with any object in the list. 
        
        Parameters
        ----------
        ray : Ray
            The ray to test for intersections.
        t_min : float, optional
            Minimum t value to consider for intersections. Default is 0.001.
        t_max : float, optional
            Maximum t value to consider for intersections. Default is 10e8.

        Returns
        -------
        is_hit : bool
            True if the ray hits any object, False otherwise.
        hit_point : ti.Vector
            The point of intersection if a hit occurs.
        hit_point_normal : ti.Vector
            The normal at the point of intersection if a hit occurs.
        front_face : bool
            True if the ray hits the front face of the object, False otherwise.
        material : int
            The material type of the hit object.
        color : ti.Vector
            The color of the hit object.        
        """
        closest_t = t_max
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        color = ti.Vector([0.0, 0.0, 0.0])
        material = 1
        for index in ti.static(range(len(self.objects))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].hit(ray, t_min, closest_t)
            if is_hit_tmp:
                closest_t = root_tmp
                is_hit = is_hit_tmp
                hit_point = hit_point_tmp
                hit_point_normal = hit_point_normal_tmp
                front_face = front_face_tmp
                material = material_tmp
                color = color_tmp
        return is_hit, hit_point, hit_point_normal, front_face, material, color

    @ti.func
    def hit_shadow(self, ray, t_min=0.001, t_max=10e8):
        is_hit_source = False
        is_hit_source_temp = False
        hitted_dielectric_num = 0
        is_hitted_non_dielectric = False
        # Compute the t_max to light source
        is_hit_tmp, root_light_source, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = \
        self.objects[0].hit(ray, t_min)
        for index in ti.static(range(len(self.objects))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].hit(ray, t_min, root_light_source)
            if is_hit_tmp:
                if material_tmp != 3 and material_tmp != 0:
                    is_hitted_non_dielectric = True
                if material_tmp == 3:
                    hitted_dielectric_num += 1
                if material_tmp == 0:
                    is_hit_source_temp = True
        if is_hit_source_temp and (not is_hitted_non_dielectric) and hitted_dielectric_num == 0:
            is_hit_source = True
        return is_hit_source, hitted_dielectric_num, is_hitted_non_dielectric


@ti.data_oriented
class Camera:
    """Camera used to generate rays in a 3D scene.

    Parameters
    ----------
    fov : float, optional
        Horizontal field of view in degrees. Default is 60.
    aspect_ratio : float, optional
        Aspect ratio of the camera (width / height). Default is 1.0.

    Attributes
    ----------
    lookfrom : ti.Vector.field
        The position of the camera in world space.
    lookat : ti.Vector.field
        The point the camera is looking at.
    vup : ti.Vector.field
        The up direction of the camera. It should be roughly orthogonal to
        the vector from ``lookfrom`` to ``lookat``; any non-orthogonal
        component is removed during reset.
    fov : float
        Horizontal field of view in degrees.
    aspect_ratio : float
        Aspect ratio of the camera (width / height).
    cam_lower_left_corner : ti.Vector.field
        The lower-left corner of the camera's image plane.
    cam_horizontal : ti.Vector.field
        The horizontal span of the camera's image plane.
    cam_vertical : ti.Vector.field
        The vertical span of the camera's image plane.
    cam_origin : ti.Vector.field
        The origin point of the camera (ray origin).
    """

    def __init__(self, fov: float = 60.0, aspect_ratio: float = 1.0):
        # Camera parameters (see class docstring for details).
        self.lookfrom = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.vup = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.fov = fov
        self.aspect_ratio = aspect_ratio

        self.cam_lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_vertical = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_origin = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.reset()

    @ti.kernel
    def reset(self):
        """Reset the camera to a default pose and recompute derived vectors."""
        self.lookfrom[None] = [0.0, 1.0, -5.0]
        self.lookat[None] = [0.0, 1.0, -1.0]
        self.vup[None] = [0.0, 1.0, 0.0]

        theta = self.fov * (PI / 180.0)
        half_height = ti.tan(theta / 2.0)
        half_width = self.aspect_ratio * half_height

        self.cam_origin[None] = self.lookfrom[None]
        w = (self.lookfrom[None] - self.lookat[None]).normalized()
        u = (self.vup[None].cross(w)).normalized()
        v = w.cross(u)

        self.cam_lower_left_corner[None] = (
            self.cam_origin[None] - w - half_width * u - half_height * v
        )
        self.cam_horizontal[None] = 2 * half_width * u
        self.cam_vertical[None] = 2 * half_height * v

    @ti.func
    def get_ray(self, u: float, v: float) -> Ray:
        """Generate a ray from the camera through the image plane.

        Parameters
        ----------
        u : float
            Horizontal coordinate on the image plane in ``[0, 1]``.
        v : float
            Vertical coordinate on the image plane in ``[0, 1]``.

        Returns
        -------
        Ray
            Ray starting at ``cam_origin`` and passing through the
            corresponding point on the image plane.
        """
        return Ray(
            self.cam_origin[None],
            self.cam_lower_left_corner[None]
            + u * self.cam_horizontal[None]
            + v * self.cam_vertical[None]
            - self.cam_origin[None],
        )
