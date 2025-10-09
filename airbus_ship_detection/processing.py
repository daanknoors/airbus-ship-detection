import numpy as np
import polars as pl

def process_labels(df):
    """
    Process the dataframe to create a new column 'n_ships' that contains the number of ships in each image.
    """
    # count number of ships per image, show 0 if if encoded_pixels is null
    df = df.with_columns(
        pl.when(pl.col('EncodedPixels').is_null())
        .then(0)
        .otherwise(pl.col('ImageId').count().over('ImageId'))
        .alias('n_ships')
    )
    # mask has ship
    df = df.with_columns(
        (pl.col('n_ships') > 0).cast(pl.Boolean).alias('has_ship')
    )
    return df
 

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    source of code: https://www.kaggle.com/code/paulorzp/run-length-encode-and-decode
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
