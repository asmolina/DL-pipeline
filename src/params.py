### Korsak
# Get table with types and ages
# Target map shape is: (h, w) = (12775, 6965)

### Nevelsk
# Get table with types and ages
# Target map shape is: (h, w) = (10461, 2379)
#### -------------------------------> [x]
####|          ____________
####|         |            |
####|         |   TRAIN    |
####|         |            |
####|   (0, a).____________.(w, a)
####|         |            |
####|         |    TEST    |
####|         |            |
####|   (h, a).____________.(w, h)
####|
####|
####\/ [y]

from rasterio.windows import Window

h, w = 10461, 2379
a = 8000

y0, x0 = 0, 0
train_window = Window.from_slices((y0, y0 + a), (x0, x0 + w))

y0, x0 = a, 0
test_window = Window.from_slices((y0, h), (x0, x0 + w))


batch_size = 5

path_S2B = './data/images/nevelsk/20180806'
channels_list = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
target = '1layer_group_2groups'

