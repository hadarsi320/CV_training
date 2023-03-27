import json

from shapes import *


class ShapeFactory:
    def create_shapes_from_list(self, list_of_shapes):
        shapes = []
        factory = ShapeFactory()
        for shape_dict in list_of_shapes:
            shapes.append(factory.create_shape(**shape_dict))
        return shapes

    def create_shape(self, type, **kwargs):
        if type == 'basic':
            shape = self.create_basic_shape(**kwargs)
        elif type == 'composite':
            shape = self.create_composite_shape(**kwargs)
        else:
            raise ValueError(f'Unknown shape type given {type}')
        return shape

    def create_basic_shape(self, shape, **kwargs):
        shape = shape.lower()
        if shape == 'line':
            shape_obj = Line(**kwargs)
        elif shape == 'circle':
            shape_obj = Circle(**kwargs)
        elif shape == 'point':
            shape_obj = Point(**kwargs)
        elif shape == 'triangle':
            shape_obj = Triangle(**kwargs)
        elif shape == 'rectangle':
            shape_obj = Rectangle(**kwargs)
        else:
            raise ValueError(f'Unknown shape: {shape}')
        return shape_obj

    def create_composite_shape(self, **kwargs):
        if 'shapes' in kwargs:
            shapes = self.create_shapes_from_list(kwargs.pop('shapes'))
        elif 'json' in kwargs:
            list_of_shapes = json.load(open(kwargs.pop('json')))
            shapes = self.create_shapes_from_list(list_of_shapes)
        else:
            raise ValueError('Composite must have either a "shapes" or a "json" field')
        composite = Composite(shapes, **kwargs)
        return composite
