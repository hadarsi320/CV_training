import json

from shape_factory import ShapeFactory

__all__ = ['create_shapes_from_json']


def create_shapes_from_json(path):
    list_of_shapes = json.load(open(path))
    factory = ShapeFactory()
    shapes = factory.create_shapes_from_list(list_of_shapes)
    return shapes
