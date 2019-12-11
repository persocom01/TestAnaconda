# Demonstrates using matplotlib to display images.
# Useful for displaying wordclouds.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pleiades as ple
from nltk.corpus import stopwords
from PIL import Image
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator

# Demonstrates opening and displaying a basic image file.
with open('./datasets/Innocence.jpg', 'rb') as f:
    img = plt.imread(f)

fig, ax = plt.subplots(figsize=(12.5, 7.5))
# ax.imshow(X, cmap=None, norm=None, aspect=None, interpolation=None,
# alpha=None, vmin=None, vmax=None, origin=None, extent=None, filternorm=1,
# filterrad=4.0, resample=None, url=None, *, data=None, **kwargs)
# interpolation appears to determine how fuzzy the pixels are going to be. See
# here for details:
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/interpolation_methods.html
ax.imshow(img)
ax.axis('off')
plt.show()
plt.close()

# Demonstrates generation of a wordcloud.
import_path = r'.\datasets\reddit.csv'
data = pd.read_csv(import_path)
df = data[['title', 'subreddit']]

X = df['title']
y = df['subreddit'].values

reddit_lingo = {
    'TIL': '',
    '[tT]oday [iI] [lL]earned': '',
    'ff+uu+': 'ffuuu'
}

cz = ple.CZ()
X = cz.text_list_cleaner(X, cz.contractions, reddit_lingo, r'[^a-zA-Z ]', cz.lemmatize_sentence)

# Join the text into a single body.
full_text = ' '.join(X)
stops = stopwords.words('english') + ['wa', 'ha']
print(stops)

import_path = r'.\datasets\Reddit.jpg'
# The way masks work is all values that are 255 in the matrix are treated as
# off limits to the wordcloud. It is possible to manipulate the mask using
# functions if you wish to reverse the image or something.
mask = np.array(Image.open(import_path))
print(mask[0])
# Generates color based on an image.
image_colors = ImageColorGenerator(mask)
# WordCloud(font_path=None, width=400, height=200, margin=2, ranks_only=None,
# prefer_horizontal=0.9, mask=None, scale=1, color_func=None, max_words=200,
# min_font_size=4, stopwords=None, random_state=None, background_color='black',
# max_font_size=None, font_step=1, mode='RGB', relative_scaling='auto',
# regexp=None, collocations=True, colormap=None, normalize_plurals=True,
# contour_width=0, contour_color='black', repeat=False, include_numbers=False,
# min_word_length=0)
# max_font_size, max_words, and background_color are the primary arguments used
# to manipulate the wordcloud.
# contour_width and contour_color are used to create an outline to the cloud.
# background_color=None and mode='RGBA' at the same time makes the background
# transparent.
# stopwords=None does not mean stopwords will not be removed. It actually means
# that the default in-built stopwords list will be used. To keep stopwords in
# the wordcloud, pass an empty list.
cloud = WordCloud(background_color='white', max_words=200,
                  mask=mask, stopwords=stops, mode='RGB')
cloud.generate(full_text)
fig, ax = plt.subplots(figsize=(12.5, 7.5))
# Recoloring the wordcloud is done in this step. If a default wordcloud is
# desired, pass cloud without recolor.
# ax.imshow(cloud, interpolation='bilinear')
ax.imshow(cloud.recolor(color_func=image_colors), interpolation='bilinear')
ax.axis('off')
plt.show()
plt.close()

# Demonstrates exporting the wordcloud to a file.
export_path = r'.\datasets\wordcloud.jpg'
cloud.to_file(export_path)
