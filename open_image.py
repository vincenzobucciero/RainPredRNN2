from PIL import Image
import matplotlib.pyplot as plt

im = Image.open("/storage/external_01/hiwefi/data/rdr0_val_previews/epoch_000/predictions/frame_01_pred.tiff")
plt.imshow(im, cmap='gray')

plt.show()
