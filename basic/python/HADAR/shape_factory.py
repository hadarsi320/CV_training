from copy import deepcopy

import json
import shapes
from shapes import Composite


class ShapeFactory:
    def __init__(self):
        self.composite_shapes = {}

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
        try:
            shape_class = getattr(shapes, shape)
        except AttributeError:
            raise ValueError(f'Unknown shape: {shape}')
        shape_obj = shape_class(**kwargs)
        return shape_obj

    def create_composite_shape(self, **kwargs):
        if 'shapes' in kwargs:
            shapes = self.create_shapes_from_list(kwargs.pop('shapes'))
        elif 'json' in kwargs:
            json_file = kwargs.pop('json')
            if json_file in self.composite_shapes:
                shapes = deepcopy(self.composite_shapes[json_file])
            else:
                list_of_shapes = json.load(open(json_file))
                shapes = self.create_shapes_from_list(list_of_shapes)
                self.composite_shapes[json_file] = deepcopy(shapes)
                print('Created composite shape')
        else:
            raise ValueError('Composite must have either a "shapes" or a "json" field')
        composite = Composite(shapes, **kwargs)
        return composite
