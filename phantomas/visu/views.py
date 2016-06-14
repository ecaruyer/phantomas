"""
Views (in ``vtk``) for fibers and isotropic regions.
"""
import vtk
from matplotlib.colors import colorConverter
import numpy as np

major_version = vtk.vtkVersion.GetVTKMajorVersion()

class ViewFiber():
    """
    A fiber is represented as a vtkPolyData, and attached to a vtk renderer.
    """
    def __init__(self, fiber, ren, **kwargs):
        self.init_draw(fiber, ren, **kwargs)
        self.draw(fiber)


    def init_draw(self, fiber, ren, **kwargs):
        # Convert color to rgb.
        my_color = kwargs.get('color', 'b')
        label = kwargs.get('label', '')
        rgb_color = 255 * colorConverter.to_rgba(my_color)

        # Create a vtkPoints instance.
        points = vtk.vtkPoints()
        self.points = points
        poly_line = vtk.vtkPolyLine()
        nb_points = fiber.get_nb_points()
        poly_line.GetPointIds().SetNumberOfIds(nb_points)
        for i in range(nb_points):
            poly_line.GetPointIds().SetId(i, i)
            self.points.InsertNextPoint(fiber.get_points()[i])
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(poly_line)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(cells)

        # The tube is wrapped around the generated streamline.
        stream_tube = vtk.vtkTubeFilter()
        self.tube = stream_tube
        if major_version <= 5:
            stream_tube.SetInput(poly_data)
        else:
            stream_tube.SetInputData(poly_data)
        stream_tube.SetRadius(fiber.get_radius())
        stream_tube.SetNumberOfSides(12)
        stream_tube.SetCapping(True)
        map_stream_tube = vtk.vtkPolyDataMapper()
        map_stream_tube.SetInputConnection(stream_tube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(map_stream_tube)
        actor.GetProperty().SetColor(rgb_color[0], rgb_color[1], rgb_color[2])
        ren.AddActor(actor)
        self.actor = actor

        # Add label to the figure
        text_source = vtk.vtkVectorText()
        text_source.SetText(label)
        text_source.Update()
        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text_source.GetOutputPort())         

        text_actor1 = vtk.vtkActor()
        text_actor1.SetMapper(text_mapper)
        text_actor1.GetProperty().SetColor(rgb_color[0], 
                                           rgb_color[1], 
                                           rgb_color[2])
        text_actor1.SetScale(1.0)
        center1 = text_actor1.GetCenter()
        text_actor1.RotateX(90)
        text_actor1.AddPosition(fiber.get_points()[-1] * 1.2 - center1)

        text_actor2 = vtk.vtkActor()
        text_actor2.SetMapper(text_mapper)
        text_actor2.GetProperty().SetColor(rgb_color[0], 
                                           rgb_color[1], 
                                           rgb_color[2])
        text_actor2.SetScale(1.0)
        center2 = text_actor2.GetCenter()
        text_actor2.RotateX(90)
        text_actor2.AddPosition(fiber.get_points()[0] * 1.2 - center2)

        ren.AddActor(text_actor1)
        ren.AddActor(text_actor2)


    def get_actor(self):
        return self.actor


    def draw(self, fiber):
        nb_points = fiber.get_nb_points()
        for i in range(nb_points):
            self.points.SetPoint(i, fiber.get_points()[i])

        self.actor.GetProperty().SetOpacity(1.0)

        # The tube is wrapped around the generated streamline.
        self.tube.SetRadius(fiber.get_radius())


    def notify(self, fiber):
        """
        If the internal structure of a fiber has changed, we need to modify 
        the vtk objects accordingly."""
        self.draw(fiber)


class ViewIsotropicRegion():
    """
    An isotropic region (sphere) is represented as a vtkPolyData, and attached 
    to a vtk renderer.
    """
    def __init__(self, isotropic_region, ren, **kwargs):
        self.init_draw(isotropic_region.center, isotropic_region.radius, ren, 
                       **kwargs)


    def init_draw(self, center, radius, ren, **kwargs):
        # Convert color to rgb.
        my_color = kwargs.get('color', 'b')
        label = kwargs.get('label', '')
        rgb_color = 255 * colorConverter.to_rgba(my_color)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(40)

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(rgb_color[0], 
                                            rgb_color[1], 
                                            rgb_color[2])

        transform = vtk.vtkTransform()
        transform.Translate(center[0], center[1], center[2])
        sphere_actor.SetUserMatrix(transform.GetMatrix())

        ren.AddActor(sphere_actor)
        self.actor = sphere_actor
  
        # Add label to the figure
        text_source = vtk.vtkVectorText()
        text_source.SetText(label)
        text_source.Update()
        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text_source.GetOutputPort())         
        text_actor1 = vtk.vtkActor()
        text_actor1.SetMapper(text_mapper)
        text_actor1.GetProperty().SetColor(rgb_color[0], 
                                           rgb_color[1], 
                                           rgb_color[2])
        text_actor1.SetScale(0.05)
        center1 = text_actor1.GetCenter()
        text_actor1.RotateX(90)
        text_actor1.AddPosition(center)
        text_actor1.AddPosition([0., 0., radius + 0.1])
        ren.AddActor(text_actor1)


    def get_actor(self):
        return self.actor


class ViewSphere():
    """
    Simply adds a translucent sphere (representing cortical surface).
    """
    def __init__(self, radius, ren, **kwargs):
        self.init_draw(radius, ren, **kwargs)


    def init_draw(self, radius, ren, **kwargs):
        # Convert color to rgb.
        opacity = kwargs.get('opacity', 0.5)
        rgb_color = [0.9, 0.9, 1.0]

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(20)
        sphere.SetPhiResolution(40)

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(rgb_color[0], 
                                            rgb_color[1], 
                                            rgb_color[2])
        sphere_actor.GetProperty().SetOpacity(opacity)

        ren.AddActor(sphere_actor)
