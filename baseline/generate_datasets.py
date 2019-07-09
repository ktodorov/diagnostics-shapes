import numpy as np
import cairo
from data.image import Image
from enums.image_property import ImageProperty
from string import ascii_lowercase

import os
import pickle

N_CELLS = 3

WIDTH = 30
HEIGHT = 30

CELL_WIDTH = WIDTH / N_CELLS
CELL_HEIGHT = HEIGHT / N_CELLS
N_CHANNELS = 3

BIG_RADIUS = CELL_WIDTH * 0.75 / 2
SMALL_RADIUS = CELL_WIDTH * 0.5 / 2

SHAPE_CIRCLE = 0
SHAPE_SQUARE = 1
SHAPE_TRIANGLE = 2
N_SHAPES = SHAPE_TRIANGLE + 1

SIZE_SMALL = 0
SIZE_BIG = 1
N_SIZES = SIZE_BIG + 1

COLOR_RED = 0
COLOR_GREEN = 1
COLOR_BLUE = 2
N_COLORS = COLOR_BLUE + 1


def draw(shape, color, size, left, top, ctx):
    center_x = (left + 0.5) * CELL_WIDTH
    center_y = (top + 0.5) * CELL_HEIGHT

    radius = SMALL_RADIUS if size == SIZE_SMALL else BIG_RADIUS
    radius *= 0.9 + np.random.random() * 0.2

    if color == COLOR_RED:
        rgb = np.asarray([1.0, 0.0, 0.0])
    elif color == COLOR_GREEN:
        rgb = np.asarray([0.0, 1.0, 0.0])
    else:
        rgb = np.asarray([0.0, 0.0, 1.0])
    rgb += np.random.random(size=(3,)) * 0.4 - 0.2
    rgb = np.clip(rgb, 0.0, 1.0)

    if shape == SHAPE_CIRCLE:
        ctx.arc(center_x, center_y, radius, 0, 2 * np.pi)
    elif shape == SHAPE_SQUARE:
        ctx.new_path()
        ctx.move_to(center_x - radius, center_y - radius)
        ctx.line_to(center_x + radius, center_y - radius)
        ctx.line_to(center_x + radius, center_y + radius)
        ctx.line_to(center_x - radius, center_y + radius)
    else:
        ctx.new_path()
        ctx.move_to(center_x - radius, center_y + radius)
        ctx.line_to(center_x, center_y - radius)
        ctx.line_to(center_x + radius, center_y + radius)
    ctx.set_source_rgb(*rgb)
    ctx.fill()

def get_target_image(
        seed,
        horizontal_position,
        vertical_position,
        shape,
        color,
        size):

    # np.random.seed(seed)

    data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)
    PIXEL_SCALE = 2
    surf = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(0.0, 0.0, 0.0)
    ctx.paint()

    shapes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
    colors = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
    sizes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]

    shapes[horizontal_position][vertical_position] = shape
    colors[horizontal_position][vertical_position] = color
    sizes[horizontal_position][vertical_position] = size

    draw(shapes[horizontal_position][vertical_position],
        colors[horizontal_position][vertical_position],
        sizes[horizontal_position][vertical_position],
        vertical_position,
        horizontal_position,
        ctx)

    metadata = {"shapes": shapes, "colors": colors, "sizes": sizes}

    return Image(shapes, colors, sizes, data, metadata)


def generate_image(
        seed,
        horizontal_position,
        vertical_position,
        shape,
        color,
        size,
        property_to_change: ImageProperty):
    np.random.seed(seed)

    target_image = get_target_image(
        seed,
        horizontal_position,
        vertical_position,
        shape,
        color,
        size)
    target_image.data = target_image.data[:, :, 0:3]

    if property_to_change == ImageProperty.Shape:
        n = N_SHAPES - 1
    elif property_to_change == ImageProperty.Size:
        n = N_SIZES - 1
    elif property_to_change == ImageProperty.Color:
        n = N_COLORS - 1
    else:
        n = N_CELLS - 1

    new_horizontal_position = horizontal_position
    new_vertical_position = vertical_position
    new_shape = shape
    new_color = color
    new_size = size

    result_images = []
    value_to_change = 0

    for _ in range(n):
        data = np.zeros((WIDTH, HEIGHT, 4), dtype=np.uint8)
        surf = cairo.ImageSurface.create_for_data(
            data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
        ctx = cairo.Context(surf)
        ctx.set_source_rgb(0.0, 0.0, 0.0)
        ctx.paint()
        
        shapes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
        colors = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]
        sizes = [[None for c in range(N_CELLS)] for r in range(N_CELLS)]

        if property_to_change == ImageProperty.Shape:
            if value_to_change == shape:
                value_to_change += 1

            new_shape = value_to_change
        elif property_to_change == ImageProperty.Size:
            if value_to_change == size:
                value_to_change += 1

            new_size = value_to_change
        elif property_to_change == ImageProperty.Color:
            if value_to_change == color:
                value_to_change += 1

            new_color = value_to_change
        elif property_to_change == ImageProperty.HorizontalPosition:
            if value_to_change == horizontal_position:
                value_to_change += 1

            new_horizontal_position = value_to_change
        elif property_to_change == ImageProperty.VerticalPosition:
            if value_to_change == vertical_position:
                value_to_change += 1

            new_vertical_position = value_to_change

        # Random location
        shapes[new_horizontal_position][new_vertical_position] = new_shape
        colors[new_horizontal_position][new_vertical_position] = new_color
        sizes[new_horizontal_position][new_vertical_position] = new_size

        draw(shapes[new_horizontal_position][new_vertical_position],
             colors[new_horizontal_position][new_vertical_position],
             sizes[new_horizontal_position][new_vertical_position],
             new_vertical_position,
             new_horizontal_position,
             ctx)
        
        value_to_change += 1

        metadata = {"shapes": shapes, "colors": colors, "sizes": sizes}
        new_image = Image(shapes, colors, sizes, data, metadata)
        new_image.data = new_image.data[:, :, 0:3]
        result_images.append(new_image)

    return target_image, result_images

def get_random_set(target_images, all_images):
    co = str(np.random.randint(3))
    ro = str(np.random.randint(3))
    sh = str(np.random.randint(3))
    co = str(np.random.randint(3))
    si = str(np.random.randint(2))

    pr = str(np.random.randint(5))
    target = co+ro+sh+co+si+pr
    print(target, all_images[target])
    return target_images[target], all_images[target]

def generate_step3_dataset(randomized=False):
    print('Running step 3 - generating dataset')

    # From Serhii's original experiment
    train_size = 74504
    valid_size = 8279
    test_size = 40504

    id_symbols = [c+b+a for c in ascii_lowercase for b in ascii_lowercase for a in ascii_lowercase]

    np.random.seed(42)

    save_step3_datasets(train_size, id_symbols, 'train', randomized, True)
    save_step3_datasets(train_size, id_symbols, 'train', randomized, True)
    save_step3_datasets(train_size, id_symbols, 'train', randomized, True)
    save_step3_datasets(train_size, id_symbols, 'train', randomized, True)
    save_step3_datasets(train_size, id_symbols, 'train', randomized, True)
    save_step3_datasets(valid_size, id_symbols, 'valid', randomized)
    save_step3_datasets(test_size, id_symbols, 'test', randomized)

def save_step3_datasets(n, id_symbols, dataset, randomized=False, zero_shot_bool=False):
    ids = 0
    all_targets = {}
    all_distractors = {}
    while n > 0:
        ids += 1
        if randomized:
            target_imgs, distractor_imgs = generate_property_set_randomizer(id_symbols[ids])
        else:
            target_imgs, distractor_imgs = generate_property_set(id_symbols[ids])
        n -= len(target_imgs)
        all_targets.update(target_imgs)
        all_distractors.update(distractor_imgs)

    if randomized:
        randomized_add = 'RANDOM'
    else:
        randomized_add = ''

    zero_shot_name = ''
    if zero_shot_bool:
        print(zero_shot_name, len(all_targets))

        all_targets, all_distractors, zero_shot_name, valid_target_images, valid_distractor_images = zero_shot_iteration(all_targets, all_distractors, zero_shot_name)
        # all_targets, all_distractors, zero_shot_name = zero_shot_iteration(all_targets, all_distractors, zero_shot_name)
        # all_targets, all_distractors, zero_shot_name = zero_shot_iteration(all_targets, all_distractors, zero_shot_name)

        print(zero_shot_name, len(all_targets))
        pickle_target = open(f'data/step3/target_dict.valid{randomized_add}.{zero_shot_name}.p','wb')
        pickle.dump(valid_target_images, pickle_target)
        pickle_target.close()

        pickle_distractors = open(f'data/step3/distractor_dict.valid{randomized_add}.{zero_shot_name}.p','wb')
        pickle.dump(valid_distractor_images, pickle_distractors)
        pickle_distractors.close()

    if not os.path.exists('data/step3'):
        os.makedirs('data/step3')

    print(f'Save {dataset} dataset for target/distractor dictionaries')
    
    pickle_target = open(f'data/step3/target_dict.{dataset}{randomized_add}.{zero_shot_name}.p','wb')
    pickle.dump(all_targets, pickle_target)
    pickle_target.close()

    pickle_distractors = open(f'data/step3/distractor_dict.{dataset}{randomized_add}.{zero_shot_name}.p','wb')
    pickle.dump(all_distractors, pickle_distractors)
    pickle_distractors.close()


def generate_property_set(id_symbol):
    image_properties = list(map(int, ImageProperty))
    seed = 42

    all_images = {}
    target_images = {}

    for shape in range(N_SHAPES):
        for size in range(N_SIZES):
            for color in range(N_COLORS):
                for horizontal_position in range(N_CELLS):
                    for vertical_position in range(N_CELLS):
                        for image_property in image_properties:
                            target_image, current_images = generate_image(seed, horizontal_position, vertical_position, shape, color, size, image_property)
                            all_images[f'{horizontal_position}{vertical_position}{shape}{color}{size}{image_property}{id_symbol}'] = current_images
                            target_images[f'{horizontal_position}{vertical_position}{shape}{color}{size}{image_property}{id_symbol}'] = target_image
    
    return target_images, all_images

def generate_property_set_randomizer(id_symbol, lower_h = 0, lower_v = 0, lower_sh = 0, lower_c = 0, lower_si = 0):
    seed = 42
    num_distractors = 3

    distractor_images = {}
    target_images = {}

    distractor_set = []

    n = N_CELLS*N_CELLS*N_SHAPES*N_COLORS*N_SIZES

    target_h = np.random.randint(lower_h, N_CELLS,size=n)
    target_v = np.random.randint(lower_v, N_CELLS,size=n)
    target_shape = np.random.randint(lower_sh, N_SHAPES,size=n)
    target_color = np.random.randint(lower_c, N_COLORS,size=n)
    target_size = np.random.randint(lower_si, N_SIZES,size=n)

    distract_h = np.random.randint(lower_h, N_CELLS,size=(num_distractors,n))
    distract_v = np.random.randint(lower_v, N_CELLS,size=(num_distractors,n))
    distract_shape = np.random.randint(lower_sh, N_SHAPES,size=(num_distractors,n))
    distract_color = np.random.randint(lower_c, N_COLORS,size=(num_distractors,n))
    distract_size = np.random.randint(lower_si, N_SIZES,size=(num_distractors,n))

    random_property_placeholder = np.random.randint(5,size=n)

    for i in range(n):
        # pass_bool = zero_shot_check(target_h[i],target_v[i],target_shape[i],target_color[i],target_size[i])
        # if pass_bool:
        target = get_target_image(
            seed,
            target_h[i],
            target_v[i],
            target_shape[i],
            target_color[i],
            target_size[i])
        target.data = target.data[:, :, 0:3]
        target_name = f'{target_h[i]}{target_v[i]}{target_shape[i]}{target_color[i]}{target_size[i]}{random_property_placeholder[i]}{id_symbol}'

        distractor_set = []
        for j in range(num_distractors):
            distractor = get_target_image(
                seed,
                distract_h[j,i],
                distract_v[j,i],
                distract_shape[j,i],
                distract_color[j,i],
                distract_size[j,i])
            distractor.data = distractor.data[:, :, 0:3]
            distractor_set.append(distractor)
        distractor_images[target_name] = distractor_set
        target_images[target_name] = target

    return target_images, distractor_images

# zero_shot_check is hardcoded for simplicity.
# def zero_shot_check(h, v, shape, color, size):
#     if h == np.random.randint(3) and v == np.random.randint(3)
#         return False
#     elif h == np.random.randint(3) and shape == np.random.randint(3)
#         return False
#     elif h == np.random.randint(3) and color == np.random.randint(3)
#         return False
#     elif h == np.random.randint(3) and size == np.random.randint(3)
#         return False
#     elif v == np.random.randint(3) and shape == np.random.randint(3)
#         return False
#     elif v == np.random.randint(3) and color == np.random.randint(3)
#         return False
#     elif v == np.random.randint(3) and size == np.random.randint(3)
#         return False
#     elif shape == np.random.randint(3) and color == np.random.randint(3)
#         return False
#     elif shape == np.random.randint(3) and size == np.random.randint(3)
#         return False
#     elif color == np.random.randint(3) and size == np.random.randint(3)
#         return False
#     return True

def zero_shot_removal(target_images, distractor_images):
    rand_prop = np.random.randint(5,size=2)
    while rand_prop[0] == rand_prop[1]:
        rand_prop = np.random.randint(5,size=2)
    rand_val = np.random.randint(3,size=2)
    while rand_prop[0] == 4 and rand_val[0] == 2:
        rand_val = np.random.randint(3,size=2)
    while rand_prop[1] == 4 and rand_val[1] == 2:
        rand_val = np.random.randint(3,size=2)
    zero_shot = np.vstack((rand_prop, rand_val))

    a = [k for k in list(target_images.keys()) if int(k[zero_shot[0,0]]) == zero_shot[1,0] and int(k[zero_shot[0,1]]) == zero_shot[1,1]]

    print(len(target_images))
    valid_target_images = {}
    valid_distractor_images = {}
    for key in a:
        valid_target_images[key] = target_images[key]
        valid_distractor_images[key] = distractor_images[key] 
        del target_images[key]
        del distractor_images[key]
        # print(key, key[zero_shot[0,0]], key[zero_shot[0,1]])
    print(zero_shot_translator(zero_shot))

    print(len(target_images), len(valid_target_images), len(target_images)+len(valid_target_images))

    # for key in list(target_images.keys()):
    #     print(key)
    # crash

    return target_images, distractor_images, zero_shot, valid_target_images, valid_distractor_images

def zero_shot_translator(zero_shot):
    props = ['horizontal','vertical','shape','color','size']
    prop1 = zero_shot[0,0]
    prop2 = zero_shot[0,1]
    val1 = zero_shot[1,0]
    val2 = zero_shot[1,1]

    return props[prop1]+str(val1)+'_'+props[prop2]+str(val2)

def zero_shot_iteration(target_images, distractor_images, zero_shot_name):
    target_images, distractor_images, zero_shot, valid_target_images, valid_distractor_images = zero_shot_removal(target_images, distractor_images)
    zero_shot_addition = zero_shot_translator(zero_shot)
    zero_shot_name += zero_shot_addition + '_'
    return target_images, distractor_images, zero_shot_name, valid_target_images, valid_distractor_images


if __name__ == "__main__":
    # target_images, distractor_images = generate_property_set_randomizer('aab')

    # print(list(target_images.keys()))
    # print(len(target_images))

    # zero_shot_name = ''
    # target_images, distractor_images, zero_shot = zero_shot_removal(target_images, distractor_images)
    # zero_shot_addition = zero_shot_translator(zero_shot)
    # zero_shot_name += zero_shot_addition + '_'
    # # print(len(target_images), len(distractor_images))
    # print(zero_shot_translator(zero_shot))

    # target_images, distractor_images, zero_shot = zero_shot_removal(target_images, distractor_images)
    # zero_shot_addition = zero_shot_translator(zero_shot)
    # zero_shot_name += zero_shot_addition + '_'
    # # print(len(target_images), len(distractor_images))
    # print(zero_shot_translator(zero_shot))

    # target_images, distractor_images, zero_shot = zero_shot_removal(target_images, distractor_images)
    # zero_shot_addition = zero_shot_translator(zero_shot)
    # zero_shot_name += zero_shot_addition
    # print(len(target_images), len(distractor_images))
    # print(zero_shot_translator(zero_shot))

    # print(zero_shot_name)
    # print(distractor_images)

    generate_step3_dataset(True)
    # target_images, all_images = generate_property_set_randomizer('zz')

    # print(target_images)
    # print(all_images)

    # pickle_target = open(f'data/target_dict_{str(len(all_images))}.p', 'rb')
    # target_dict = pickle.load(pickle_target)

    # pickle_distractors = open(f'data/distractor_dict_{str(len(all_images))}.p', 'rb')
    # distractors_dict = pickle.load(pickle_distractors)

    # target, distractors = get_random_set(target_dict, distractors_dict)
