from drawer import Drawer
from json_reader import create_shapes_from_json


def main():
    shapes = create_shapes_from_json('json/magnun_opus.json')
    drawer = Drawer()
    drawer.draw(shapes, [900, 1600], [225, 225, 225])
    drawer.save('magnum_opus.png')
    drawer.show()


if __name__ == '__main__':
    main()
